from __future__ import annotations
from pathlib import Path
from typing import List
import logging
import os

import numpy as np
import pyrosetta
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.core.scoring import fa_rep
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.loops import get_fa_scorefxn
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.core.pack.task import TaskFactory

from poli.core.abstract_isolated_function import AbstractIsolatedFunction

pyrosetta.init()

ROSETTA_ENERGY_DIR = Path(__file__).parent.resolve()


class RosettaEnergyIsolatedLogic(AbstractIsolatedFunction):
    """
    RosettaEnergy internal implementation.

    Parameters
    ----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s), by default None.
    score_function : str, optional
        Which Rosetta score function to use. Options are [ref2015 , default , centroid , fa].
        The default function references ref2015.
    seeds : int, optional
        Overwrite Rosetta random seed with own integer, uses mt19937 RT reference (as per Rosetta default).
    unit : str, optional
        Output unit of black-box. Default is delta REU, which is difference between variant and wild-type.
        Alternatives are: 
            REU -- raw energy function computed value, 
            DDG -- scaled delta REU
    conversion_factor : float, optional
        Scaling factor applied when computing DDGs (DDG=deltaREU/conversion_factor). Only applies to DDG conversion. 
        Default is '2.9' following:
            Park, Hahnbeom, et al. 
            "Simultaneous optimization of biomolecular energy functions on features from small molecules and macromolecules." 
            Journal of chemical theory and computation 12.12 (2016): 6201-6212.
            DOI: https://doi.org/10.1021/acs.jctc.6b00819
    clean : bool, optional
        Flag whether to apply PDB cleaning routine prior to pose loading of reference.
        Default is True.

    Methods
    -------
    _apply_on_sequence(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the energy units of the mutant(s) in x.

    Raises
    ------
    AssertionError
        If wildtype_pdb_path is not provided.
        If the provided energy units are not valid.
    NotImplementedError
        If energy function does not exist.

    Notes
    -----
    - The wildtype_pdb_path is a single Path object.
    """
    def __init__(
        self,
        wildtype_pdb_path: Path | List[Path],
        score_function: str = "ref2015",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        n_threads: int = 4,
    ):
        # set number of threads prior to initializing Rosetta
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        pyrosetta.init(extra_options=f"-corre.pack:omp_threads {n_threads}")
        self.conversion_factor = conversion_factor  # iff ddGs computed from REUs
        self.relax = relax  # do we use FastRelax?
        self.cycle = cycle  # iterations on relax protocol
        valid_units = ["DDG", "REU", "DREU"]
        if unit.upper() not in valid_units:
            raise AssertionError(f"Output unit {unit} not a valid unit: {valid_units}")
        self.unit = unit.upper()
        if seed is not None:
            pyrosetta.rosetta.basic.random.init_random_generators(seed, "mt19937")
        self.score_function_identifier = score_function
        self.energy_function = self.__get_score_fn(score_function)
        self.energy_function.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 3)
        self.energy_function.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 5)
        if clean:
            cleanATOM(wildtype_pdb_path)
            self.pose = pyrosetta.pose_from_pdb(str(wildtype_pdb_path.with_suffix(".clean.pdb")))
        else:
            self.pose = pyrosetta.pose_from_pdb(str(wildtype_pdb_path))
        # set packing protocol
        self.pack = None
        if pack:
            pack_task = TaskFactory.create_packer_task(self.pose)
            pack_task.restrict_to_repacking()
            self.packer = PackRotamersMover(self.energy_function, pack_task)
            self.packer.apply(self.pose)
        self.relax = None
        # set relax protocol; NOTE: relax & pack required when larger/destabilizing changes are introduced
        if relax:
            self.relax = FastRelax(self.energy_function, self.cycle)
            self.relax.apply(self.pose)
        # set MinMover; NOTE: suitable for small changes to structure
        self.mover = MinMover()
        self.mover.min_type("lbfgs_armijo_nonmonotone")
        self.mover.score_function(self.energy_function)
        self.wt_score = self.energy_function.score(self.pose)
        self.wt_fa_rep = self.pose.energies().total_energies()[fa_rep]
    
    def __get_score_fn(self, score_function_identifier: str):
        if score_function_identifier == "default":
            return get_score_function(True)
        elif score_function_identifier == "ref2015":
            return get_score_function(True)
        elif score_function_identifier == "centroid":
            return get_score_function(False)
        elif score_function_identifier == "fa":
            return get_fa_scorefxn()
        else:
            raise NotImplementedError

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        y = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            seq = "".join(x[i, :].tolist())
            y[i, 0] = self._apply_on_sequence(seq)
        return y

    def _apply_on_sequence(self, seq: str) -> float:
        protein_pose = pyrosetta.pose_from_sequence(seq)
        self.mover.apply(protein_pose)
        REU = self.energy_function.score(protein_pose)
        clashes = False
        if self.relax is not None and self.pack is not None:
            clashes = self._check_pose_steric_clashes(mutant_pose=protein_pose)
        if clashes:
            logging.info("Steric clashes detected, attempt recovery by relax and pack-minimize...")
            self._apply_relax_pack(protein_pose)
            logging.info("Recompute energy function")
            REU = self.energy_function.score(protein_pose)
        if self.unit == "REU":
            return REU
        elif self.unit == "DREU":
            print(self.wt_score)
            return REU - self.wt_score
        elif self.unit == "DDG":
            print("WT")
            print(self.wt_score)
            return ( REU - self.wt_score ) / self.conversion_factor
        else:
            RuntimeError("Output unit not specified correctly!")

    def _check_pose_steric_clashes(self, mutant_pose: object) -> bool:
        mutant_fa_rep = mutant_pose.energies().total_energies()[fa_rep]
        return bool(mutant_fa_rep > self.wt_fa_rep)
    
    def _apply_relax_pack(self, mutant_pose: object) -> None:
        self.pack.apply(mutant_pose)
        self.relax.apply(mutant_pose)


if __name__ == "__main__":
    # from poli.core.registry import register_isolated_function

    # register_isolated_function(
    #     RosettaEnergyIsolatedLogic,  # Your function, designed to be isolated
    #     name=rosetta_energy_information.get_problem_name() + "__isolated",  #  Same name as the problem and folder, ending on __isolated.
    #     conda_environment_name="poli__rosetta_energy",  # The name of the conda env inside environment.yml.
    # )
    from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residue_strings

    pdb_path = Path(ROSETTA_ENERGY_DIR) / "1ggx.pdb"
    seq = parse_pdb_as_residue_strings(pdb_path)
    
    isolated_logic = RosettaEnergyIsolatedLogic(unit="DDG", wildtype_pdb_path=pdb_path)

    y = isolated_logic._apply_on_sequence("".join(seq))
    print(y)
    print("batched")
    seq_arr = np.vstack([np.array(seq) for i in range(5)])
    for i in range(seq_arr.shape[0]):
        seq_arr[i,10+i] = "A"
    y = isolated_logic(seq_arr)
    print(y)
