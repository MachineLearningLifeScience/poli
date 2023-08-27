"""
This module contains utilities for querying
foldx for repairing and simulating the mutations
of proteins. 
"""
from typing import List
from pathlib import Path
import shutil
import subprocess

from Bio.PDB.Residue import Residue
from Bio.PDB import SASA

from poli.core.util.proteins.mutations import mutations_from_wildtype_and_mutant
from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residues,
    parse_pdb_as_structure,
)

# Making sure that the foldx executable is where
# we expect
PATH_TO_FOLDX_FILES = Path().home() / "foldx"
if not PATH_TO_FOLDX_FILES.exists():
    raise FileNotFoundError(
        "Please download FoldX and place it in your home directory. \n"
        "We expect it to find the following files: \n"
        "   - the binary at: ~/foldx/foldx  \n"
        "   - the rotabase file at: ~/foldx/rotabase.txt \n"
    )

if not (PATH_TO_FOLDX_FILES / "foldx").exists():
    raise FileNotFoundError(
        "Please compile FoldX and place it in your home directory as 'foldx'. \n"
        "We expect it to find the following files: \n"
        "   - the binary at: ~/foldx/foldx  \n"
        "   - the rotabase file at: ~/foldx/rotabase.txt \n"
    )

if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
    raise FileNotFoundError(
        "Please place the rotabase.txt file in your foldx directory. "
        "We expect it to find the following paths: \n"
        "   - the binary at: ~/foldx/foldx  \n"
        "   - the rotabase file at: ~/foldx/rotabase.txt \n"
    )


class FoldxInterface:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir

    def repair(self, pdb_file: Path) -> None:
        """
        This method repairs a PDB file with FoldX.

        TODO: implement this.
        """
        raise NotImplementedError

    def _simulate_mutations(self, pdb_file: Path, mutations: List[str]) -> None:
        """
        This method simulates mutations on a PDB file with FoldX.
        The list of mutations must be as expected by the
        individual_list.txt, i.e. a list of strings of the following
        form:
            - the first letter is the original residue,
            - the second letter is the chain ID,
            - the third letter is the position of the mutation,
            - the fourth letter is the mutant residue.
        e.g. ["MA0A"] means that the first residue in the chain
        is mutated from M to A.
        """
        # TODO: add support for multiple mutations.
        assert len(mutations) == 1, "We only support single mutations for now. "

        # We load up the residues of the wildtype
        wildtype_residues = parse_pdb_as_residues(pdb_file)

        # We write the individual_list.txt in the working directory
        self.write_mutations_to_file(wildtype_residues, mutations, self.working_dir)

        # We copy the relevant files into the working directory
        self.copy_foldx_files(pdb_file)

        # We run foldx
        # This generates the Raw_*.fxout file, which
        # contains the stability. If that file wasn't
        # generated, we raise an error.
        foldx_command = [
            str(PATH_TO_FOLDX_FILES / "foldx"),
            "--command=BuildModel",
            "--pdb",
            f"{pdb_file.stem}.pdb",
            "--mutant-file",
            "individual_list.txt",
            "--water",
            "-CRYSTAL",
            "--pH",
            "7.0",
        ]
        subprocess.run(foldx_command, cwd=self.working_dir)

        results_dir = self.working_dir / f"Raw_{pdb_file.stem}.fxout"
        mutated_structure_dir = self.working_dir / f"{pdb_file.stem}_1.pdb"
        assert results_dir.exists(), (
            f"FoldX did not generate the expected results file {results_dir}. "
            f"Please check the working directory: {self.working_dir}. "
        )
        assert mutated_structure_dir.exists(), (
            f"FoldX did not generate the expected mutated pdb file {mutated_structure_dir}. "
            f"Please check the working directory: {self.working_dir}. "
        )

    def compute_stability(self, pdb_file: Path, mutations: List[str]) -> float:
        if not (self.working_dir / f"Raw_{pdb_file.stem}.fxout").exists():
            self._simulate_mutations(pdb_file, mutations)

        with open(self.working_dir / f"Raw_{pdb_file.stem}.fxout") as f:
            lines = f.readlines()

        # TODO: add support for multiple mutations
        assert len(mutations) == 1, "We only support single mutations for now. "

        # The energy is at the second to last line
        # and in the second column.
        energy = float(lines[-2].split()[1])
        return -energy

    def compute_sasa(self, pdb_file: Path, mutations: List[str]) -> float:
        if not (self.working_dir / f"Raw_{pdb_file.stem}.fxout").exists():
            self._simulate_mutations(pdb_file, mutations)

        # Loading up the mutation's pdb file
        mutated_structure = parse_pdb_as_structure(
            self.working_dir / f"{pdb_file.stem}_1.pdb", structure_name="pdb_mutated"
        )
        sasa_computer = SASA.ShrakeRupley()

        # This computes the sasa score, and attaches
        # it to the structure.
        sasa_computer.compute(mutated_structure, level="S")

        return mutated_structure.sasa

    def copy_foldx_files(self, pdb_file: Path):
        """
        We copy the rotabase and pdb file to the working directory.
        """
        shutil.copy(
            PATH_TO_FOLDX_FILES / "rotabase.txt", self.working_dir / "rotabase.txt"
        )
        shutil.copy(pdb_file, self.working_dir / f"{pdb_file.stem}.pdb")

    @staticmethod
    def write_mutations_to_file(
        wildtype_resiudes: List[Residue], mutations: List[str], output_dir: Path
    ) -> None:
        """
        This method writes the list of mutations to a file
        in the given directory.
        """
        # Write the mutations in the format of individual_list.txt
        lines = []
        for mutation in mutations:
            # e.g. mutations_in_line = ["MA0A", "AA5G"]
            # would represent two mutations of the same
            # wildtype: one at residue 0, which is mutated
            # from M to A; and one at residue 5, which is
            # mutated from A to G.
            # The A in between is the chain ID.
            mutations_in_line = mutations_from_wildtype_and_mutant(
                wildtype_resiudes, mutation
            )

            # These are represented as a single line, separated
            # by commas, and terminated by a semicolon.
            lines.append(",".join(mutations_in_line) + ";")

        with open(output_dir / "individual_list.txt", "w") as f:
            f.writelines(lines)
