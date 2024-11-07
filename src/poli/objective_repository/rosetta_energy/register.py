"""
This module implements Rosetta-energy Stability Predictions,
using PyRosetta-4 [1].

Rosetta is a black box that starts
with one wildtype pdb file, and assesses the
stability of a variant.
Two general running-modes are available: 
i) fast prediction with MinMover and scoring function -- set relax=False, pack=False
ii) default protocol prediction with pack and relax algorithm -- default .

[1] Chaudhury, Sidhartha, Sergey Lyskov, and Jeffrey J. Gray. 
"PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta." 
Bioinformatics 26.5 (2010): 689-691. https://doi.org/10.1093/bioinformatics/btq007 

"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.objective_repository.rosetta_energy.information import (
    rosetta_energy_information,
)

CONSENT_FILE = Path(__file__).parent.resolve() / ".pyrosetta_accept.txt"


def has_opted_in(consent_file: Path = CONSENT_FILE) -> bool:
    if consent_file.exists():
        with open(consent_file, "r") as file:
            consent_status = file.read().strip()
            return consent_status == "accepted"
    return False


def opt_in_wrapper(f: Callable, *args, **kwargs):
    if not has_opted_in():
        agreement = input(
            "I have read and accept the License Agreements of PyRosetta, subject to the Rosetta™ license. ([Y]es/[N]o) \n See https://www.pyrosetta.org/home/licensing-pyrosetta and https://els2.comotion.uw.edu/product/rosetta ."
        )
        if agreement.strip().lower() == "yes" or agreement.strip().lower() == "y":
            with open(CONSENT_FILE, "w") as file:
                file.write("accepted")
            return f
        else:
            print(
                "You must accept and be in compliance with the original PyRosetta, Rosetta™ license."
            )
            raise RuntimeError
    else:
        return f


class RosettaEnergyBlackBox(AbstractBlackBox):
    """
    RosettaEnergy Black Box implementation.

    Parameters
    ----------
    wildtype_pdb_path : Path
        The path to the wildtype PDB file(s), by default None.
        NOTE: currently only for single PDB files implemented -- List of Paths not yet supported!
    score_function : str, optional
        Which Rosetta score function to use. Options are [ref2015 , default , centroid , fa, ref2015_cart, franklin2019].
        The default function references ref2015.
    seed : int, optional
        Overwrite Rosetta random seed with own integer, uses mt19937 RT reference (as per Rosetta default).
    unit : str, optional
        Output unit of black-box. Default is DDG, which is scaled difference between variant and wild-type.
        Alternatives are:
            REU -- raw energy function value,
            DREU -- energy unit delta to wild-type.
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
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the stability of the mutant(s) in x.
    get_black_box_info()
        Returns BlackBox info object.

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
        score_function: str = "default",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        constraint_weight: float = 5.0,
        n_threads: int = 4,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        assert wildtype_pdb_path is not None
        self.force_isolation = force_isolation
        self.wildtype_pdb_path = wildtype_pdb_path
        self.score_function = score_function
        self.seed = seed
        self.unit = unit
        self.conversion_factor = conversion_factor
        self.clean = clean
        self.relax = relax
        self.pack = pack
        self.cycle = cycle
        self.constraint_weight = constraint_weight
        self.n_threads = n_threads

        inner_function = get_inner_function(
            isolated_function_name="rosetta_energy__isolated",
            class_name="RosettaEnergyIsolatedLogic",
            module_to_import="poli.objective_repository.rosetta_energy.isolated_function",
            force_isolation=self.force_isolation,
            wildtype_pdb_path=self.wildtype_pdb_path,
            score_function=self.score_function,
            seed=self.seed,
            unit=self.unit,
            conversion_factor=self.conversion_factor,
            clean=self.clean,
            relax=self.relax,
            pack=self.pack,
            cycle=self.cycle,
            constraint_weight=self.constraint_weight,
            n_threads=self.n_threads,
        )
        self.inner_function = opt_in_wrapper(inner_function)
        self.x0 = self.inner_function.x0

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Computes the stability of the mutant(s) in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            The stability(DDGs/REUs/DREUs) of the mutant(s) in x.
        """
        return self.inner_function(x, context)

    def get_black_box_info(self) -> BlackBoxInformation:
        """
        Returns the black box information for Rosetta.
        """
        is_fixed_length = True
        max_sequence_length = len("".join(list(self.x0[0])))
        return BlackBoxInformation(
            name="rosetta_energy",
            max_sequence_length=max_sequence_length,
            aligned=False,
            fixed_length=is_fixed_length,
            deterministic=False,
            alphabet=AMINO_ACIDS,
            log_transform_recommended=False,
            discrete=True,
            fidelity="high",
            padding_token="",
        )


class RosettaEnergyProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        return rosetta_energy_information

    def create(
        self,
        wildtype_pdb_path: Path | List[Path],
        score_function: str = "default",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        constraint_weight: int | float = 5,
        n_threads: int = 4,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a Rosetta black box instance, alongside initial
        observations.

        Parameters
        ----------
        wildtype_pdb_path : Path
            The path to the wildtype PDB file(s), by default None.
            NOTE: currently only for single PDB files implemented -- List of Paths not yet supported!
        score_function : str, optional
            Which Rosetta score function to use. Options are [ref2015 , default , centroid , fa, ref2015_cart, franklin2019].
            The default function references ref2015.
        seed : int, optional
            Overwrite Rosetta random seed with own integer, uses mt19937 RT reference (as per Rosetta default).
        unit : str, optional
            Output unit of black-box. Default is DDG, which is scaled difference between variant and wild-type.
            Alternatives are:
                REU -- raw energy function value,
                DREU -- energy unit delta to wild-type.
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
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The evaluation budget, by default float("inf").

        Returns
        -------
        Problem : poli.core.problem.Problem
            The poli Problem instance, containing f , x0 , y0
            where
            f : RosettaEnergyBlackBox
                The Rosetta black box instance.
            x0 : np.ndarray
                The initial observations (i.e. the wildtype as sequences
                of amino acids).
            y0 : np.ndarray
                The initial observations (i.e. the stability of the wildtype).
        """
        # Creating your black box function
        f = RosettaEnergyBlackBox(
            wildtype_pdb_path=wildtype_pdb_path,
            score_function=score_function,
            seed=seed,
            unit=unit,
            conversion_factor=conversion_factor,
            clean=clean,
            relax=relax,
            pack=pack,
            cycle=cycle,
            constraint_weight=constraint_weight,
            n_threads=n_threads,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Your first input (an np.array[str] of shape [b, L] or [b,])
        x0 = f.inner_function.x0

        return Problem(f, x0)
