"""
This script registers the stability FoldX black box and objective factory.

FoldX [1] is a simulator that allows for computing the difference
in free energy between a wildtype protein and a mutated protein. We
also use biopython for pre-processing the PDB files [2].

[1] Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F.,
    & Serrano, L. (2005). The FoldX web server: an online force field.
    Nucleic acids research, 33(suppl_2), W382-W388.
[2] Cock PA, Antao T, Chang JT, Chapman BA, Cox CJ, Dalke A, Friedberg I,
    Hamelryck T, Kauff F, Wilczynski B and de Hoon MJL (2009) Biopython:
    freely available Python tools for computational molecular biology and
    bioinformatics. Bioinformatics, 25, 1422-1423
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import FoldXNotFoundException
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch


class FoldXStabilityBlackBox(AbstractBlackBox):
    """
    A black box implementation for evaluating the stability of protein structures using FoldX.

    Parameters
    ----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s).
    experiment_id : str, optional
        The ID of the experiment (default is None).
    tmp_folder : Path, optional
        The path to the temporary folder (default is None).
    eager_repair : bool, optional
        Whether to eagerly repair the protein structures (default is False).
    verbose : bool, optional
        Whether to print the output from FoldX (default is False).
    batch_size : int, optional
        The batch size for parallel processing (default is None).
    parallelize : bool, optional
        Whether to parallelize the computation (default is False).
    num_workers : int, optional
        The number of workers for parallel processing (default is None).
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Methods
    -------
    _black_box(x, context)
        Runs the given input x and pdb files provided in the context through FoldX and returns the total stability score.

    Notes
    -----
    We expect the user to have FoldX v5.0 installed and compiled.
    More specifically, we expect a binary file called foldx to be
    in the path ~/foldx/foldx.
    """

    def __init__(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        verbose: bool = False,
        batch_size: int = 1,
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
        self.wildtype_pdb_path = wildtype_pdb_path
        self.experiment_id = experiment_id
        self.tmp_folder = tmp_folder
        self.eager_repair = eager_repair
        self.verbose = verbose
        self.force_isolation = force_isolation

        if not (Path.home() / "foldx" / "foldx").exists():
            raise FoldXNotFoundException(
                "FoldX wasn't found in ~/foldx/foldx. Please install it."
            )

        inner_function = get_inner_function(
            isolated_function_name="foldx_stability__isolated",
            class_name="FoldXStabilityIsolatedLogic",
            module_to_import="poli.objective_repository.foldx_stability.isolated_function",
            force_isolation=force_isolation,
            wildtype_pdb_path=wildtype_pdb_path,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
        )
        self.x0 = inner_function.x0
        self.wildtype_amino_acids = inner_function.wildtype_amino_acids

    def _black_box(self, x: np.ndarray, context: None) -> np.ndarray:
        """
        Runs the given input x and pdb files provided
        in the context through FoldX and returns the
        total energy score.

        Since the goal is MINIMIZING the energy,
        we return the negative of the total energy
        (a.k.a. the stability).

        Parameters
        ----------
        x : np.ndarray
            The input array representing the mutations.
        context : None
            The context for the computation.

        Returns
        -------
        np.ndarray
            The array of stability scores.

        """
        inner_function = get_inner_function(
            isolated_function_name="foldx_stability__isolated",
            class_name="FoldXStabilityIsolatedLogic",
            module_to_import="poli.objective_repository.foldx_stability.isolated_function",
            force_isolation=self.force_isolation,
            quiet=True,
            wildtype_pdb_path=self.wildtype_pdb_path,
            experiment_id=self.experiment_id,
            tmp_folder=self.tmp_folder,
            eager_repair=self.eager_repair,
            verbose=self.verbose,
        )
        return inner_function(x, context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="foldx_stability",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=AMINO_ACIDS,
            log_transform_recommended=False,
            discrete=True,
            fidelity=None,
            padding_token="",
        )


class FoldXStabilityProblemFactory(AbstractProblemFactory):
    def create(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        verbose: bool = False,
        seed: int = None,
        batch_size: int = 1,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a FoldX stability black box function and initial observations.

        Parameters
        ----------
        wildtype_pdb_path : Union[Path, List[Path]]
            Path(s) to the wildtype PDB file(s).
        alphabet : List[str], optional
            List of amino acids to use as the alphabet.
        experiment_id : str, optional
            Identifier for the experiment.
        tmp_folder : Path, optional
            Path to the temporary folder for storing intermediate files.
        eager_repair : bool, optional
            Whether to eagerly repair the protein structures.
        verbose : bool, optional
            Whether to print the output from FoldX.
        seed : int, optional
            Seed for random number generation.
        batch_size : int, optional
            Number of sequences to process in parallel.
        parallelize : bool, optional
            Whether to parallelize the computation.
        num_workers : int, optional
            Number of worker processes to use for parallel computation.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem : Problem
            An instance of a FoldX stability problem.

        Raises
        ------
        ValueError
            If wildtype_pdb_path is missing or has an invalid type.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if wildtype_pdb_path is None:
            raise ValueError(
                "Missing required argument wildtype_pdb_path. "
                "Did you forget to pass it to create()?"
            )

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = [Path(wildtype_pdb_path.strip())]
        elif isinstance(wildtype_pdb_path, Path):
            wildtype_pdb_path = [wildtype_pdb_path]
        elif isinstance(wildtype_pdb_path, list):
            if isinstance(wildtype_pdb_path[0], str):
                wildtype_pdb_path = [Path(x.strip()) for x in wildtype_pdb_path]
            elif isinstance(wildtype_pdb_path[0], Path):
                pass
        else:
            raise ValueError(
                f"wildtype_pdb_path must be a string or a Path. Received {type(wildtype_pdb_path)}"
            )
        # By this point, we know that wildtype_pdb_path is a
        # list of Path objects.

        # TODO: add support for a larger batch-size.
        f = FoldXStabilityBlackBox(
            wildtype_pdb_path=wildtype_pdb_path,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        wildtype_amino_acids_ = f.wildtype_amino_acids

        longest_wildtype_length = max([len(x) for x in wildtype_amino_acids_])

        wildtype_amino_acids = [
            amino_acids + [""] * (longest_wildtype_length - len(amino_acids))
            for amino_acids in wildtype_amino_acids_
        ]

        x0 = np.array(wildtype_amino_acids).reshape(
            len(wildtype_pdb_path), longest_wildtype_length
        )

        problem = Problem(f, x0)

        return problem
