"""
Registers the stability and SASA FoldX black box
and objective factory.

FoldX [1] is a simulator that allows for computing the difference
in free energy between a wildtype protein and a mutated protein.
We pair this with biopython [2] to compute the SASA of the mutated
protein.

[1] Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F.,
    & Serrano, L. (2005). The FoldX web server: an online force field.
    Nucleic acids research, 33(suppl_2), W382-W388.
[2] Cock PA, Antao T, Chang JT, Chapman BA, Cox CJ, Dalke A, Friedberg I,
    Hamelryck T, Kauff F, Wilczynski B and de Hoon MJL (2009) Biopython:
    freely available Python tools for computational molecular biology and
    bioinformatics. Bioinformatics, 25, 1422-1423
"""

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


class FoldXStabilityAndSASABlackBox(AbstractBlackBox):
    """
    A black box implementation for computing the solvent accessible surface area (SASA) score using FoldX.

    Parameters
    -----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s).
    experiment_id : str, optional
        The ID of the experiment. Default is None.
    tmp_folder : Path, optional
        The path to the temporary folder. Default is None.
    eager_repair : bool, optional
        Whether to perform eager repair. Default is False.
    verbose : bool, optional
        Whether to print the output from FoldX. Default is False.
    batch_size : int, optional
        The batch size for parallel processing. Default is None.
    parallelize : bool, optional
        Whether to parallelize the computation. Default is False.
    num_workers : int, optional
        The number of workers for parallel processing. Default is None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

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
            isolated_function_name="foldx_stability_and_sasa__isolated",
            class_name="FoldXStabilitityAndSASAIsolatedLogic",
            module_to_import="poli.objective_repository.foldx_stability_and_sasa.isolated_function",
            wildtype_pdb_path=wildtype_pdb_path,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
        )
        self.wildtype_amino_acids = inner_function.wildtype_amino_acids

    def _black_box(self, x: np.ndarray, context: None) -> np.ndarray:
        """
        Runs the given input x and pdb files provided
        in the context through FoldX and returns the
        total energy score.

        Since the goal is MINIMIZING the energy,
        we return the negative of the total energy.

        Parameters
        -----------
        x : np.ndarray
            The input array representing the mutations.
        context : None
            The context for the black box computation.

        Returns
        --------
        y: np.ndarray
            The computed stability and SASA score(s) as a numpy array.
        """
        inner_function = get_inner_function(
            isolated_function_name="foldx_stability_and_sasa__isolated",
            class_name="FoldXStabilitityAndSASAIsolatedLogic",
            module_to_import="poli.objective_repository.foldx_stability_and_sasa.isolated_function",
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
            name="foldx_stability_and_sasa",
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


class FoldXStabilityAndSASAProblemFactory(AbstractProblemFactory):
    """
    Factory class for creating FoldX SASA (Solvent Accessible Surface Area) problems.

    Methods
    -------
    create:
        Creates a problem instance with the specified parameters.
    """

    def create(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        verbose: bool = False,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Create a FoldXSASABlackBox object and compute the initial values of wildtypes.

        Parameters
        ----------
        wildtype_pdb_path : Union[Path, List[Path]]
            Path or list of paths to the wildtype PDB files.
        experiment_id : str, optional
            Identifier for the experiment.
        tmp_folder : Path, optional
            Path to the temporary folder for intermediate files.
        eager_repair : bool, optional
            Flag indicating whether to perform eager repair.
        verbose : bool, optional
            Flag indicating whether to print the output from FoldX.
        seed : int, optional
            Seed for random number generators. If None is passed,
            the seeding doesn't take place.
        batch_size : int, optional
            Number of samples per batch for parallel computation.
        parallelize : bool, optional
            Flag indicating whether to parallelize the computation.
        num_workers : int, optional
            Number of worker processes for parallel computation.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem : Problem
            A problem instance containing a FoldXStabilityAndSASABlackBox
            function, and initial wildtypes x0.

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

        # We use the default alphabet if None was provided.
        # See ENCODING in foldx_utils.py
        f = FoldXStabilityAndSASABlackBox(
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

        # We need to compute the initial values of all wildtypes
        # in wildtype_pdb_path. For this, we need to specify x0,
        # a vector of wildtype sequences. These are padded to
        # match the maximum length with empty strings.
        wildtype_amino_acids_ = f.wildtype_amino_acids

        # # We may need to parallelize. If so, the inner_function object
        # # can't be pickled. If that's the case, we need to delete it.
        # # Each call to the black box will create its own connection
        # # to an isolated function.
        # del f.inner_function

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
