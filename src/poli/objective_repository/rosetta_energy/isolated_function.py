from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pose import deep_copy
from pyrosetta.rosetta.core.scoring import (
    ScoreFunctionFactory,
    fa_rep,
    get_score_function,
)
from pyrosetta.rosetta.protocols.loops import get_fa_scorefxn
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import cleanATOM, mutate_residue

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residue_strings

pyrosetta.init()

ROSETTA_ENERGY_DIR = Path(__file__).parent.resolve()


class RosettaEnergyIsolatedLogic(AbstractIsolatedFunction):
    """
    RosettaEnergy internal implementation.

    Parameters
    ----------
    wildtype_pdb_path : Path
        The path to the wildtype PDB file(s), by default None.
        NOTE: currently only for single PDB files implemented -- List of Paths not yet supported!
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
    relax: bool, optional
        Flag whether to apply relaxation protocol. Default enabled.
        NOTE: disable if fast compute required
    pack: bool, optional
        Flag whether to apply packing protocol. Default enabled.
        NOTE: disable if fast compute required
    cycle: int, optional
        Number of relaxation cycles applied.
    constraint_weight: int, optional
        Constraint on scoring function for atom-pair, angles, coordinates.

    Methods
    -------
    _apply_on_sequence(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the energy units of the mutant(s) in x.
    _check_pose_steric_clashes(mutant_pose)
        Checks for steric clashes of variant pose against WT pose, returns boolean.
    _apply_relax_pack(mutant_pose)
        Apply pack and relax routines on pose.

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
        wildtype_pdb_path: Path,
        score_function: str = "default",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        constraint_weight: float | int = 5,
        n_threads: int = 4,
    ):
        # set number of threads prior to initializing Rosetta
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        pyrosetta.init(extra_options=f"-nthreads {n_threads}")
        self.conversion_factor = conversion_factor  # iff ddGs computed from REUs
        self.relax = relax  # do we use FastRelax?
        self.cycle = cycle  # iterations on relax protocol
        self.constraint_weight = constraint_weight
        valid_units = ["DDG", "REU", "DREU"]
        if unit.upper() not in valid_units:
            raise AssertionError(f"Output unit {unit} not a valid unit: {valid_units}")
        self.unit = unit.upper()
        if seed is not None:
            pyrosetta.rosetta.basic.random.init_random_generators(seed, "mt19937")
        self.score_function_identifier = score_function
        self.energy_function = self.__get_score_fn(score_function)
        self.energy_function.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint,
            self.constraint_weight,
        )
        self.energy_function.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint,
            self.constraint_weight,
        )
        self.energy_function.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint,
            self.constraint_weight,
        )
        self.x0 = np.array([parse_pdb_as_residue_strings(wildtype_pdb_path)])
        if not isinstance(wildtype_pdb_path, Path):
            raise TypeError(f"{wildtype_pdb_path} is not a Path")
        if clean:
            cleanATOM(wildtype_pdb_path)
            self.pose = pyrosetta.pose_from_pdb(
                str(wildtype_pdb_path.with_suffix(".clean.pdb"))
            )
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
        # set MinMover; NOTE: suitable for small changes to structure
        self.mover = MinMover()
        # self.mover.min_type("lbfgs_armijo_nonmonotone")
        self.mover.score_function(self.energy_function)
        # set relax protocol; NOTE: relax & pack required when larger/destabilizing changes are introduced
        if relax:
            self.relax = FastRelax(self.energy_function, self.cycle)
            cartesian_scoring = bool("cart" in self.score_function_identifier)
            self.relax.cartesian(cartesian_scoring)
            self.relax.minimize_bond_angles(cartesian_scoring)
            self.relax.minimize_bond_lengths(cartesian_scoring)
            self.relax.apply(self.pose)
        self.wt_score = self.energy_function.score(self.pose)
        self.wt_fa_rep = self.pose.energies().total_energies()[fa_rep]
        self.x_t = None  # track sequences as property

    def __get_score_fn(self, score_function_identifier: str):
        if score_function_identifier == "default":
            return get_score_function(True)
        elif score_function_identifier == "ref2015":
            return get_score_function(True)
        elif score_function_identifier == "ref2015_cart":
            return ScoreFunctionFactory.create_score_function(score_function_identifier)
        elif score_function_identifier == "centroid":
            return get_score_function(False)
        elif score_function_identifier == "franklin2019":
            # TODO: requires AddMembraneMover , not yet functional!
            return ScoreFunctionFactory.create_score_function(score_function_identifier)
        elif score_function_identifier == "fa":
            return get_fa_scorefxn()
        else:
            raise NotImplementedError("Invalid scoring function!")

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        self.x_t = []
        y = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            y[i, 0] = self._apply_on_sequence(x[i, :])
        return y

    def _apply_on_sequence(self, seq_arr: str | np.ndarray) -> float:
        if not isinstance(seq_arr, np.ndarray):
            seq_arr = np.array(list(seq_arr))
        seq_arr = np.atleast_2d(seq_arr)
        pose_mutant = deep_copy(self.pose)
        diff_rosetta_idx = (
            np.where(self.x0 != seq_arr)[1] + 1
        )  # get diff indices (Rosetta indexes at 1)
        diff_residues = seq_arr[self.x0 != seq_arr]  # get residue mutant diff to WT
        for idx, res in zip(diff_rosetta_idx, diff_residues):
            mutate_residue(pose_mutant, idx, res)  # inplace mutation on pose copy
        self.x_t.append(pose_mutant.sequence())
        self.mover.apply(pose_mutant)
        REU = self.energy_function.score(pose_mutant)
        clashes = False
        if self.relax is not None and self.pack is not None:
            clashes = self._check_pose_steric_clashes(mutant_pose=pose_mutant)
        if clashes:
            logging.info(
                "Steric clashes detected, attempt recovery by relax and pack-minimize..."
            )
            self._apply_relax_pack(pose_mutant)
            logging.info("Recompute energy function")
            REU = self.energy_function.score(pose_mutant)
        if self.unit == "REU":
            return REU
        elif self.unit == "DREU":
            return REU - self.wt_score
        elif self.unit == "DDG":
            return (REU - self.wt_score) / self.conversion_factor
        else:
            RuntimeError("Output unit not specified correctly!")

    def _check_pose_steric_clashes(self, mutant_pose: object) -> bool:
        mutant_fa_rep = mutant_pose.energies().total_energies()[fa_rep]
        return bool(mutant_fa_rep > self.wt_fa_rep)

    def _apply_relax_pack(self, mutant_pose: object) -> None:
        self.pack.apply(mutant_pose)
        self.relax.apply(mutant_pose)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RosettaEnergyIsolatedLogic,
        name="rosetta_energy__isolated",
        conda_environment_name="poli__rosetta_energy",
        force=True,
    )
