"""
This module contains utilities for querying
foldx for repairing and simulating the mutations
of proteins. 
"""
from typing import List
from pathlib import Path
import shutil
import subprocess
import os

from Bio.PDB.Residue import Residue
from Bio.PDB import SASA
from Bio.SeqUtils import seq1

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

    def repair(self, pdb_file: Path, remove_and_rename: bool = False) -> None:
        """
        This method repairs a PDB file with FoldX, overwriting
        the original file if remove_and_rename is True (default: False).
        """
        # Make sure the relevant files are in the
        # working directory
        self.copy_foldx_files(pdb_file)

        # The command to run
        command = [
            str(PATH_TO_FOLDX_FILES / "foldx"),
            "--command=RepairPDB",
            "--pdb",
            f"{pdb_file.stem}.pdb",
        ]

        # Running it in the working directory
        subprocess.run(command, cwd=self.working_dir)

        # Checking that the file was generated
        repaired_pdb_file = self.working_dir / f"{pdb_file.stem}_Repair.pdb"
        assert repaired_pdb_file.exists(), (
            f"FoldX did not generate the expected repaired pdb file {repaired_pdb_file}. "
            f"Please check the working directory: {self.working_dir}. "
        )

        # Removing the old pdb file, and renaming the repaired one
        if remove_and_rename:
            shutil.rmtree(self.working_dir / f"{pdb_file.stem}.pdb")
            shutil.move(
                self.working_dir / f"{pdb_file.stem}_Repair.pdb",
                self.working_dir / f"{pdb_file.stem}.pdb",
            )

    def _simulate_mutations(self, pdb_file: Path, mutations: List[str] = None) -> None:
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
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. "

        # We load up the residues of the wildtype
        wildtype_residues = parse_pdb_as_residues(pdb_file)

        # If there are no mutations, we still need to pass
        # a dummy mutation to foldx, so we "mutate" the first
        # residue in the wildtype string. This can be achieved
        # by passing the wildtype string as the mutated string.
        if mutations is None or len(mutations) == 0:
            wildtype_string = [seq1(res.get_resname()) for res in wildtype_residues]
            mutations = ["".join(wildtype_string)]

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

    def _read_energy(self, pdb_file: Path) -> float:
        """
        This method reads the energy from a FoldX results file,
        assuming that there was a single mutation.
        TODO: add support for multiple mutations.
        """
        assert (
            self.working_dir / f"Raw_{pdb_file.stem}.fxout"
        ).exists(), f"FoldX did not generate the expected results file {self.working_dir / f'Raw_{pdb_file.stem}.fxout'}. "
        with open(self.working_dir / f"Raw_{pdb_file.stem}.fxout") as f:
            lines = f.readlines()

        energy = float(lines[-2].split()[1])
        return energy

    def _compute_sasa(self, pdb_file: Path) -> float:
        """
        This method computes the SASA from a FoldX results file,
        assuming that there was a single mutation.
        """
        print("Parsing mutated structure")
        mutated_structure = parse_pdb_as_structure(
            self.working_dir / f"{pdb_file.stem}_1.pdb", structure_name="pdb_mutated"
        )

        sasa_computer = SASA.ShrakeRupley()

        # This computes the sasa score, and attaches
        # it to the structure.
        print("Computing the internal SASA")
        sasa_computer.compute(mutated_structure, level="S")

        return mutated_structure.sasa

    def compute_stability(self, pdb_file: Path, mutations: List[str] = None) -> float:
        # if not (self.working_dir / f"Raw_{pdb_file.stem}.fxout").exists():
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. Pass no mutations if you want to compute the energy of the wildtype."

        self._simulate_mutations(pdb_file, mutations)

        stability = -self._read_energy(pdb_file)
        return stability

    def compute_sasa(self, pdb_file: Path, mutations: List[str] = None) -> float:
        # if not (self.working_dir / f"Raw_{pdb_file.stem}.fxout").exists():
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. Pass no mutations if you want to compute the SASA of the wildtype."
        self._simulate_mutations(pdb_file, mutations)

        sasa_score = self._compute_sasa(pdb_file)
        return sasa_score

    def compute_stability_and_sasa(self, pdb_file: Path, mutations: List[str] = None):
        """
        This function computes stability and sasa with a single foldx run,
        instead of two separate runs.
        """
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. Pass no mutations if you want to compute the energy and SASA of the wildtype."
        self._simulate_mutations(pdb_file, mutations)

        print("Reading stability")
        stability = -self._read_energy(pdb_file)
        print("Computing sasa")
        sasa_score = self._compute_sasa(pdb_file)

        print("Done!")
        return stability, sasa_score

    def copy_foldx_files(self, pdb_file: Path):
        """
        We copy the rotabase and pdb file to the working directory.
        """
        if not (self.working_dir / "rotabase.txt").exists():
            os.symlink(
                str(PATH_TO_FOLDX_FILES / "rotabase.txt"),
                str(self.working_dir / "rotabase.txt"),
            )
        destination_path_for_pdb = self.working_dir / f"{pdb_file.stem}.pdb"
        if not destination_path_for_pdb.exists():
            shutil.copy(pdb_file, destination_path_for_pdb)

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
