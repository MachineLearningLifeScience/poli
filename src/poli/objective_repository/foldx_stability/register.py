"""
This script registers FoldX stability as an objective function.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from time import time
from uuid import uuid4

import numpy as np

from Bio import PDB
from Bio.SeqUtils import seq1

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residue_strings,
    parse_pdb_as_residues,
)
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.proteins.mutations import mutations_from_wildtype_and_mutant
from poli.core.util.proteins.foldx import FoldxInterface

# This is the folder where all the files
# generated by FoldX will be stored.
# Feel free to change it if you want
# to keep the files somewhere else.
# An alternative is e.g. TMP_PATH = THIS_DIR
# TODO: what happens if the user is on Windows?
# TMP_PATH = THIS_DIR / "tmp"
TMP_PATH = Path("/tmp").resolve()


class FoldXStabilityBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = 1,
        wildtype_pdb_file: Path = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
    ):
        """
        TODO: document
        """
        # WARNING: notice how the batch-size is set to 1.
        # This is because we only support simulating one
        # mutation at a time.
        # TODO: fix this using parallelization.

        # TODO: assert that wildtype_pdb_file is provided
        assert wildtype_pdb_file is not None, (
            "Missing required argument wildtype_pdb_file. "
            "Did you forget to pass it to create and into the black box?"
        )

        super().__init__(info=info, batch_size=batch_size)

        if alphabet is None:
            alphabet = info.alphabet

        self.string_to_idx = {symbol: i for i, symbol in enumerate(alphabet)}
        self.idx_to_string = {v: k for k, v in self.string_to_idx.items()}

        if isinstance(wildtype_pdb_file, str):
            wildtype_pdb_file = Path(wildtype_pdb_file.strip())

        self.wildtype_pdb_file = wildtype_pdb_file

        self.wildtype_residues = parse_pdb_as_residues(wildtype_pdb_file)
        self.wildtype_amino_acids = parse_pdb_as_residue_strings(wildtype_pdb_file)
        self.wildtype_residue_string = "".join(self.wildtype_amino_acids)

        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

    def mutations_from_wildtype(self, mutated_residue_string: str) -> List[str]:
        """
        Since foldx expects an individual_list.txt file of mutations,
        this function computes the Levenshtein distance between
        the wildtype residue string and the mutated residue string,
        keeping track of the replacements.

        This method returns a list of strings which are to be written
        in a single line of individual_list.txt.
        """
        return mutations_from_wildtype_and_mutant(
            self.wildtype_residues, mutated_residue_string
        )

    def create_working_directory(self) -> Path:
        """
        TODO: document.
        """

        working_dir = TMP_PATH / "foldx_tmp_files" / self.experiment_id
        working_dir.mkdir(exist_ok=True, parents=True)

        return working_dir

    def _black_box(self, x: np.ndarray, context: None) -> np.ndarray:
        """
        Runs the given input x and pdb files provided
        in the context through FoldX and returns the
        total energy score.

        Since the goal is MINIMIZING the energy,
        we return the negative of the total energy.

        After the initial call, let's assume that
        the subsequent calls are about mutating
        the wildtype sequence. From then onwards,
        the context should contain
        - wildtype_pdb_file
        - path_to_mutation_list

        To accomodate for the initial call, if the
        path_to_mutation_list is not provided (or
        if it's None), we assume that we're supposed
        to evaluate the energy of the wildtype sequence.
        """
        # Check if the context is valid
        # delete_working_dir = context["delete_working_dir"]
        # wildtype_pdb_file = context["wildtype_pdb_file"]
        wildtype_pdb_file = self.wildtype_pdb_file

        # Create a working directory for this function call
        working_dir = self.create_working_directory()

        # Given that x, we simply define the
        # mutations to be made as a mutation_list.txt
        # file.
        mutations_as_strings = [
            "".join([amino_acid for amino_acid in x_i]) for x_i in x
        ]

        foldx_interface = FoldxInterface(working_dir)
        stability = foldx_interface.compute_stability(
            pdb_file=wildtype_pdb_file, mutations=mutations_as_strings
        )

        return np.array([stability]).reshape(-1, 1)


class FoldXStabilityProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        """
        TODO: document
        """
        alphabet = AMINO_ACIDS

        return ProblemSetupInformation(
            name="foldx_stability",
            max_sequence_length=np.inf,
            alphabet=alphabet,
            aligned=False,
        )

    def create(
        self,
        seed: int = 0,
        wildtype_pdb_path: Path = None,
        alphabet: Dict[str, int] = None,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        L = self.get_setup_information().get_max_sequence_length()
        if wildtype_pdb_path is None:
            raise ValueError(
                "Missing required argument wildtype_pdb_path. "
                "Did you forget to pass it to create()?"
            )

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = Path(wildtype_pdb_path.strip())
        elif isinstance(wildtype_pdb_path, Path):
            pass
        else:
            raise ValueError(
                f"wildtype_pdb_path must be a string or a Path. Received {type(wildtype_pdb_path)}"
            )

        if alphabet is None:
            # We use the default alphabet.
            # See AMINO_ACIDS in foldx_utils.py
            alphabet = self.get_setup_information().get_alphabet()

        problem_info = self.get_setup_information()
        # TODO: add support for a larger batch-size.
        f = FoldXStabilityBlackBox(
            info=problem_info,
            batch_size=1,
            wildtype_pdb_file=wildtype_pdb_path,
            alphabet=alphabet,
        )

        wildtype_residues = parse_pdb_as_residues(wildtype_pdb_path)
        wildtype_amino_acids = [
            seq1(residue.get_resname())
            for residue in wildtype_residues
            if residue.get_resname() != "NA"
        ]

        x0 = np.array(wildtype_amino_acids).reshape(1, -1)

        f_0 = f(x0)

        return f, x0, f_0


if __name__ == "__main__":
    from poli.core.registry import register_problem

    foldx_problem_factory = FoldXStabilityProblemFactory()
    register_problem(
        foldx_problem_factory,
        conda_environment_name="poli__protein",
    )
