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

# This is the folder where all the files
# generated by FoldX will be stored.
# Feel free to change it if you want
# to keep the files somewhere else by
# passing tmp_folder to the black box.
# TODO: what happens if the user is on Windows?
DEFAULT_TMP_PATH = Path("/tmp").resolve()


class FoldxBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        wildtype_pdb_path: Union[Path, List[Path]] = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
    ):
        """
        TODO: Document
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
        )

        if alphabet is None:
            alphabet = info.alphabet

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = Path(wildtype_pdb_path.strip())

        if isinstance(wildtype_pdb_path, Path):
            wildtype_pdb_path = [wildtype_pdb_path]

        self.wildtype_pdb_paths = wildtype_pdb_path

        self.wildtype_resiudes = [
            parse_pdb_as_residues(pdb_file) for pdb_file in wildtype_pdb_path
        ]

        self.wildtype_amino_acids = [
            parse_pdb_as_residue_strings(pdb_file) for pdb_file in wildtype_pdb_path
        ]

        self.wildtype_residue_strings = [
            "".join(amino_acids) for amino_acids in self.wildtype_amino_acids
        ]

        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

        self.tmp_folder = tmp_folder if tmp_folder is not None else DEFAULT_TMP_PATH

    def create_working_directory(self) -> Path:
        """
        TODO: document.
        """
        sub_experiment_id = str(uuid4())[:8]

        working_dir = (
            self.tmp_folder / "foldx_tmp_files" / self.experiment_id / sub_experiment_id
        )
        working_dir.mkdir(exist_ok=True, parents=True)

        return working_dir
