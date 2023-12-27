"""This module implements the FoldX black box and objective factory.

FoldX [1] is a simulator that allows for computing the difference
in free energy between a wildtype protein and a mutated protein.

References
----------
[1] The FoldX web server: an online force field.
    Nucleic acids research Schymkowitz, J., Borg, J., Stricher,
    F., Nys, R., Rousseau, F., & Serrano, L. (2005). ,
    33(suppl_2), W382-W388.
"""

from typing import Union, List
from pathlib import Path
from time import time
from uuid import uuid4

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residue_strings,
    parse_pdb_as_residues,
)
from poli.core.util.proteins.foldx import FoldxInterface

# This is the folder where all the files
# generated by FoldX will be stored.
# Feel free to change it if you want
# to keep the files somewhere else by
# passing tmp_folder to the black box.
# TODO: what happens if the user is on Windows?
DEFAULT_TMP_PATH = Path("/tmp").resolve()


class FoldxBlackBox(AbstractBlackBox):
    """
    A class representing the FoldxBlackBox, which is used for simulating protein mutations using FoldX.

    Parameters
    -----------
    info : ProblemSetupInformation, required
        The problem setup information object. (default: None)
    batch_size : int, optional
        The batch size for parallelization. (default: None)
    parallelize : bool, optional
        Flag indicating whether to parallelize the simulations. (default: False)
    num_workers : int, optional
        The number of workers for parallelization. (default: None)
    wildtype_pdb_path : Union[Path, List[Path]], required
        The path(s) to the wildtype PDB file(s). (default: None)
    alphabet : List[str], optional
        The list of allowed amino acids. (default: None)
    experiment_id : str, optional
        The experiment ID. (default: None)
    tmp_folder : Path, optional
        The temporary folder path. (default: None)
    eager_repair : bool, optional
        Flag indicating whether to eagerly repair the PDB files. (default: False)

    Attributes
    ----------
    experiment_id : str
        The experiment ID.
    tmp_folder : Path
        The temporary folder path.
    wildtype_pdb_paths : List[Path]
        The list of repaired wildtype PDB file paths.
    wildtype_residues : List[List[Residue]]
        The list of wildtype residues for each PDB file.
    wildtype_amino_acids : List[List[str]]
        The list of wildtype amino acids for each PDB file.
    wildtype_residue_strings : List[str]
        The list of wildtype residue strings for each PDB file.

    Methods
    -------
    create_working_directory() -> Path:
        Creates and returns the working directory path for the black box.

    """

    def __init__(
        self,
        info: ProblemSetupInformation = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        wildtype_pdb_path: Union[Path, List[Path]] = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
    ):
        """
        Initialize the FoldxBlackBox.

        Parameters
        -----------
        info : ProblemSetupInformation, optional
            The problem setup information object. (default: None)
        batch_size : int, optional
            The batch size for parallelization. (default: None)
        parallelize : bool, optional
            Flag indicating whether to parallelize the simulations. (default: False)
        num_workers : int, optional
            The number of workers for parallelization. (default: None)
        evaluation_budget : int, optional
            The evaluation budget. (default: float('inf'))
        wildtype_pdb_path : Union[Path, List[Path]], optional
            The path(s) to the wildtype PDB file(s). (default: None)
        alphabet : List[str], optional
            The list of allowed amino acids. (default: None)
        experiment_id : str, optional
            The experiment ID. (default: None)
        tmp_folder : Path, optional
            The temporary folder path. (default: None)
        eager_repair : bool, optional
            Flag indicating whether to eagerly repair the PDB files. (default: False)
        """
        # WARNING: notice how the batch-size is set to 1.
        # This is because we only support simulating one
        # mutation at a time.
        # TODO: fix this using parallelization.

        # TODO: assert that wildtype_pdb_file is provided
        assert wildtype_pdb_path is not None, (
            "Missing required argument wildtype_pdb_file. "
            "Did you forget to pass it to create and into the black box?"
        )
        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # Defining the experiment id
        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

        self.tmp_folder = tmp_folder if tmp_folder is not None else DEFAULT_TMP_PATH

        if alphabet is None:
            alphabet = info.alphabet

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = Path(wildtype_pdb_path.strip())

        if isinstance(wildtype_pdb_path, Path):
            wildtype_pdb_path = [wildtype_pdb_path]

        if isinstance(wildtype_pdb_path, list):
            _wildtype_pdb_path = []
            for pdb_file in wildtype_pdb_path:
                if isinstance(pdb_file, str):
                    pdb_file = Path(pdb_file.strip())
                assert isinstance(
                    pdb_file, Path
                ), f"Expected a Path object or a string, but got {type(pdb_file)}."
                _wildtype_pdb_path.append(pdb_file)

            wildtype_pdb_path = _wildtype_pdb_path

        # At this point, wildtype_pdb_path is a list of Path objects.
        # We need to ensure that these are repaired pdb files.
        # We do this by creating a temporary folder and repairing
        # the pdbs there.
        if eager_repair:
            path_for_repairing_pdbs = self.tmp_folder / "foldx_tmp_files_for_repair"
            path_for_repairing_pdbs.mkdir(exist_ok=True, parents=True)
            foldx_interface_for_repairing = FoldxInterface(path_for_repairing_pdbs)

            # Re-writing wildtype_pdb_path to be the list of repaired pdb files.
            repaired_wildtype_pdb_files = [
                foldx_interface_for_repairing._repair_if_necessary_and_provide_path(
                    pdb_file
                )
                for pdb_file in wildtype_pdb_path
            ]

            # At this point, wildtype_pdb_path is a list of Path objects.
            self.wildtype_pdb_paths = repaired_wildtype_pdb_files
        else:
            self.wildtype_pdb_paths = wildtype_pdb_path

        self.wildtype_resiudes = [
            parse_pdb_as_residues(pdb_file) for pdb_file in self.wildtype_pdb_paths
        ]

        self.wildtype_amino_acids = [
            parse_pdb_as_residue_strings(pdb_file)
            for pdb_file in self.wildtype_pdb_paths
        ]

        self.wildtype_residue_strings = [
            "".join(amino_acids) for amino_acids in self.wildtype_amino_acids
        ]

    def create_working_directory(self) -> Path:
        """
        Create and return the working directory path for the black box.

        Returns
        --------
        Path
            The path to the working directory.
        """
        sub_experiment_id = str(uuid4())[:8]

        working_dir = (
            self.tmp_folder / "foldx_tmp_files" / self.experiment_id / sub_experiment_id
        )
        working_dir.mkdir(exist_ok=True, parents=True)

        return working_dir
