"""
This module implements Rapid Stability Predictions (RaSP),
using their open-source code [1].

From a bird's eye view, RaSP is a black box that starts
with a collection of wildtype pdb files, and assesses the
stability of a (single-site) mutant. For the informed reader,
this black box can be considered a drop-in replacement of FoldX,
or Rosetta.

[1] “Rapid Protein Stability Prediction Using Deep Learning Representations.”
Blaabjerg, Lasse M, Maher M Kassem, Lydia L Good, Nicolas Jonsson,
Matteo Cagiada, Kristoffer E Johansson, Wouter Boomsma, Amelie Stein,
and Kresten Lindorff-Larsen.  Edited by José D Faraldo-Gómez,
Detlef Weigel, Nir Ben-Tal, and Julian Echave. eLife 12
(May 2023): e82593. https://doi.org/10.7554/eLife.82593.

"""

from pathlib import Path
from typing import List, Union

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.rasp.information import rasp_information


class RaspBlackBox(AbstractBlackBox):
    """
    RaSP Black Box implementation.

    Parameters
    ----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s), by default None.
    chains_to_keep : List[str], optional
        The chains to keep in the PDB file(s), by default we
        keep the chain "A" for all pdbs passed.
    experiment_id : str, optional
        The experiment ID, by default None.
    tmp_folder : Path, optional
        The temporary folder path, by default None, which means
        we will keep temporary files in /tmp.
    batch_size : int, optional
        The batch size for parallel evaluation, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize evaluation, by default False.
    num_workers : int, optional
        The number of workers for parallel evaluation, by default None.
    evaluation_budget : int, optional
        The evaluation budget, by default float("inf").

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the stability of the mutant(s) in x.
    _clean_wildtype_pdb_files()
        This function cleans the wildtype pdb files
        stored in self.wildtype_pdb_paths, using
        cached results if they exist.


    Raises
    ------
    AssertionError
        If wildtype_pdb_path is not provided.

    Notes
    -----
    - The wildtype_pdb_path can be a single Path object or a list of Path objects.
    - If chains_to_keep is not provided, it defaults to keeping chain A for all wildtypes.
    - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
    - If tmp_folder is not provided, it defaults to the default temporary path.
    """

    def __init__(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        chains_to_keep: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        """
        Initialize the RaSP Register object.

        Parameters:
        -----------
        wildtype_pdb_path : Union[Path, List[Path]]
            The path(s) to the wildtype PDB file(s).
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The evaluation budget, by default float("inf").
        chains_to_keep : List[str], optional
            The chains to keep in the PDB file(s), by default we
            keep the chain "A" for all pdbs passed.
        experiment_id : str, optional
            The experiment ID, by default None.
        tmp_folder : Path, optional
            The temporary folder path, by default None.

        Raises:
        -------
        AssertionError
            If wildtype_pdb_path is not provided.

        Notes:
        ------
        - The wildtype_pdb_path can be a single Path object or a list of Path objects.
        - If chains_to_keep is not provided, it defaults to keeping chain A for all wildtypes.
        - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
        - If tmp_folder is not provided, it defaults to the default temporary path.
        """
        assert wildtype_pdb_path is not None, (
            "Missing required argument wildtype_pdb_file. "
            "Did you forget to pass it to create and into the black box?"
        )
        if parallelize:
            print(
                "poli 🧪: RaspBlackBox parallelization is handled by the isolated logic. Disabling it."
            )
            parallelize = False
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.force_isolation = force_isolation
        self.wildtype_pdb_path = wildtype_pdb_path
        self.chains_to_keep = chains_to_keep
        self.experiment_id = experiment_id
        self.tmp_folder = tmp_folder
        self.inner_function = get_inner_function(
            isolated_function_name="rasp__isolated",
            class_name="RaspIsolatedLogic",
            module_to_import="poli.objective_repository.rasp.isolated_function",
            force_isolation=self.force_isolation,
            wildtype_pdb_path=self.wildtype_pdb_path,
            chains_to_keep=self.chains_to_keep,
            experiment_id=self.experiment_id,
            tmp_folder=self.tmp_folder,
        )
        self.x0 = self.inner_function.x0

    def _black_box(self, x, context=None):
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
            The stability of the mutant(s) in x.

        Notes
        -----
        - x is a np.array[str] of shape [b, L], where L is the length
          of the longest sequence in the batch, and b is the batch size.
          We process it by concantenating the array into a single string,
          where we assume the padding to be an empty string (if there was any).
          Each of these x_i's will be matched to the wildtype in self.  wildtype_residue_strings with the lowest Hamming distance.
        """
        return self.inner_function(x, context=context)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        """
        Returns the black box information for RaSP.
        """
        return rasp_information


class RaspProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        """
        Returns the problem setup information for RaSP.
        """
        return rasp_information

    def create(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        chains_to_keep: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a RaSP black box instance, alongside initial
        observations.

        Parameters
        ----------
        wildtype_pdb_path : Union[Path, List[Path]]
            The path(s) to the wildtype PDB file(s).
        experiment_id : str, optional
            The experiment ID, by default None.
        tmp_folder : Path, optional
            The temporary folder path, by default None.
        seed : int, optional
            The seed value for random number generation, by default None.
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
        f : RaspBlackBox
            The RaSP black box instance.
        x0 : np.ndarray
            The initial observations (i.e. the wildtypes as sequences
            of amino acids).
        y0 : np.ndarray
            The initial observations (i.e. the stability of the wildtypes).
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
                "Invalid type for wildtype_pdb_path. "
                "It must be a string, a Path, or a list of strings or Paths."
            )

        f = RaspBlackBox(
            wildtype_pdb_path=wildtype_pdb_path,
            chains_to_keep=chains_to_keep,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Constructing x0
        # (Moved to the isolated logic)
        x0 = f.inner_function.x0

        problem = Problem(f, x0)

        return problem


if __name__ == "__main__":
    from poli.core.registry import register_problem

    rasp_problem_factory = RaspProblemFactory()
    register_problem(
        rasp_problem_factory,
        conda_environment_name="poli__rasp",
        force=True,
    )
