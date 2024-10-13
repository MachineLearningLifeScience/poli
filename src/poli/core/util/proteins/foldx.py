"""FoldX interface for measuring stability and SASA.

This module implements a `FoldxInterface` for querying
foldx's repairing and simulating models [1]. This implementation
is heavily inspired by the way LaMBO uses FoldX [2].

If this module is imported from a script, it will
automatically check that the foldx files are in the
expected location, and raise an error if they are not.

References
----------
[1] The FoldX web server: an online force field.
    Schymkowitz, J., Borg, J., Stricher, F., Nys, R.,
    Rousseau, F., & Serrano, L. (2005).  Nucleic acids research,
    33(suppl_2), W382-W388.

[2] “Accelerating Bayesian Optimization for Biological Sequence
    Design withDenoising Autoencoders.” Stanton, Samuel, Wesley Maddox,
    Nate Gruver, Phillip Maffettone, Emily Delaney, Peyton Greenside,
    and Andrew Gordon Wilson.  arXiv, July 12, 2022.
    http://arxiv.org/abs/2203.12742.
    https://github.com/samuelstanton/lambo

"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from Bio.PDB import SASA
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1
from pdbtools.pdb_delhetatm import run as pdb_delhetatm_run

from poli.core.util.proteins.mutations import (
    mutations_from_wildtype_residues_and_mutant,
)
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
    )

if not (PATH_TO_FOLDX_FILES / "foldx").exists():
    raise FileNotFoundError(
        "Please compile FoldX and place it in your home directory as 'foldx'. \n"
        "We expect it to find the following files: \n"
        "   - the binary at: ~/foldx/foldx  \n"
    )


class FoldxInterface:
    """
    A class for interacting with FoldX, a protein engineering software.

    Parameters
    ----------
    working_dir : Union[Path, str]
        The working directory where FoldX files and output will be stored.

    Methods
    -------
    repair(pdb_file, remove_and_rename=False, pH=7.0, remove_heteroatoms=True)
        Repairs a PDB file with FoldX, overwriting the original file if remove_and_rename is True.
    _repair_if_necessary_and_provide_path(pdb_file)
        Repairs a PDB file if necessary and returns the path of the repaired PDB.
    _simulate_mutations(pdb_file, mutations=None)
        Simulates mutations on a PDB file with FoldX.
    _read_energy(pdb_file)
        Reads the energy from a FoldX results file.
    _compute_sasa(pdb_file)
        Computes the SASA (solvent-accessible surface area) from a FoldX results file.
    compute_stability(pdb_file, mutations=None)
        Computes the stability of a protein structure using FoldX.
    compute_sasa(pdb_file, mutations=None)
        Computes the SASA (solvent-accessible surface area) of a protein structure using FoldX.
    compute_stability_and_sasa(pdb_file, mutations=None)
        Computes the stability and SASA (solvent-accessible surface area) of a protein structure using FoldX in a single run.
    copy_foldx_files(pdb_file)
        Copies the necessary FoldX files to the working directory.
    write_mutations_to_file(wildtype_resiudes, mutations, output_dir)
        Writes the list of mutations to a file in the given directory.

    Attributes
    ----------
    working_dir : Union[Path, str]
        The working directory for FoldX.
    verbose : bool
        If True, the FoldX output will be printed to stdout.

    Notes
    -----
    This class expects you to use the binary for FoldX v.5.
    Previous versions relied on a "rotabase.txt" file, which
    is no longer used.
    """

    def __init__(self, working_dir: Union[Path, str], verbose: bool = False):
        """
        Initialize the FoldX object.

        Parameters
        ----------
        working_dir : Union[Path, str]
            The working directory for FoldX.
        verbose : bool, optional
            If True, the FoldX output will be printed to stdout. Default is False.
        """
        if isinstance(working_dir, str):
            working_dir = Path(working_dir)

        self.working_dir = working_dir
        self.verbose = verbose

        if not verbose:
            self.output = subprocess.DEVNULL
        else:
            self.output = None

    def repair(
        self,
        pdb_file: Union[str, Path],
        remove_and_rename: bool = False,
        pH: float = 7.0,
        remove_heteroatoms: bool = True,
    ) -> None:
        """
        Repairs a PDB file with FoldX, overwriting
        the original file if remove_and_rename is True (default: False).

        Parameters
        ----------
        pdb_file : Union[str, Path]
            The path to the PDB file to be repaired.
        remove_and_rename : bool, optional
            If True, the original file will be removed and the repaired file will be renamed to the original file name.
            Default is False.
        pH : float, optional
            The pH value for the repair process. Default is 7.0.
        remove_heteroatoms : bool, optional
            If True, heteroatoms will be removed from the repaired PDB file using pdbtools.
            Default is True.

        Raises
        ------
        RuntimeError
            If FoldX fails to repair the PDB file.

        Notes:
        ------
        This method repairs a PDB file using FoldX. It overwrites the original file if remove_and_rename is True.
        """
        # Make sure the pdb file is a path
        if isinstance(pdb_file, str):
            pdb_file = Path(pdb_file)

        # Make sure the relevant files are in the
        # working directory
        self.copy_foldx_files(pdb_file)

        # The command to run
        command = [
            str(PATH_TO_FOLDX_FILES / "foldx"),
            "--command=RepairPDB",
            "--pdb",
            f"{pdb_file.stem}.pdb",
            "--water",
            "-CRYSTAL",
            "--pH",
            f"{pH}",
        ]

        # Running it in the working directory
        try:
            subprocess.run(
                command,
                cwd=self.working_dir,
                check=True,
                stdout=self.output,
                stderr=self.output,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FoldX failed to repair the pdb file {pdb_file}. " + "\n"
                "This can happen because: \n"
                "- the foldx license (which has a 1-year turnaround) expired, or"
                "- the PDB file is not repaired. \n"
                "If you want to dig deeper, you can check the FoldX output in the working directory: "
                f"{self.working_dir}."
            ) from e

        # Checking that the file was generated
        repaired_pdb_file = self.working_dir / f"{pdb_file.stem}_Repair.pdb"
        assert repaired_pdb_file.exists(), (
            f"FoldX did not generate the expected repaired pdb file {repaired_pdb_file}. "
            f"Please check the working directory: {self.working_dir}. "
        )

        # If remove heteroatoms is True, we remove them
        # using pdbtools
        if remove_heteroatoms:
            # We load up the repaired file
            with open(repaired_pdb_file) as f:
                lines = f.readlines()

            deleting_heteroatoms_result = pdb_delhetatm_run(lines)

            # We write the result to the same file
            with open(repaired_pdb_file, "w") as f:
                f.writelines(deleting_heteroatoms_result)

        # Removing the old pdb file, and renaming the repaired one
        if remove_and_rename:
            shutil.rmtree(self.working_dir / f"{pdb_file.stem}.pdb")
            shutil.move(
                self.working_dir / f"{pdb_file.stem}_Repair.pdb",
                self.working_dir / f"{pdb_file.stem}.pdb",
            )

    def _repair_if_necessary_and_provide_path(self, pdb_file: Path) -> Path:
        """
        If the pdb_file's name doesn't end in "_Repair.pdb",
        then we repair it and return the path of the repaired
        pdb. Otherwise, we return the same path as the input.

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file to be repaired.

        Returns
        -------
        repaired_path: Path
            The path to the repaired PDB file.

        """
        # Make sure that we don't have a repaired pdb file
        # in the working directory (which is usually a cache)
        if (self.working_dir / f"{pdb_file.stem}_Repair.pdb").exists():
            logging.warning(
                f"Found a repaired pdb file in the cache for {pdb_file.stem}. Using it instead of repairing."
            )
            return self.working_dir / f"{pdb_file.stem}_Repair.pdb"

        # If the file's already fixed, then we don't need to
        # do anything. Else, we repair it.
        if "_Repair" in pdb_file.name:
            return pdb_file
        else:
            self.repair(pdb_file)
            return self.working_dir / f"{pdb_file.stem}_Repair.pdb"

    def _simulate_mutations(self, pdb_file: Path, mutations: List[str] = None) -> None:
        """Simulates mutations, starting from a wildtype PDB file.

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

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file to be repaired.
        mutations : List[str], optional
            The list of mutations to simulate. If None, we simulate
            the wildtype. Default is None.

        Raises
        ------
        AssertionError
            If the number of mutations is not 0 or 1.
        RuntimeError
            If FoldX fails to simulate the mutations.

        Notes:
        ------
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
        try:
            subprocess.run(
                foldx_command,
                cwd=self.working_dir,
                check=True,
                stdout=self.output,
                stderr=self.output,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FoldX failed to simulate the mutations on the pdb file {pdb_file}. "
                f"Please check the working directory: {self.working_dir}. "
            ) from e

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

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file to be repaired.

        Returns
        -------
        energy: float
            The change of energy (ddG) of the mutated structure.

        Raises
        ------
        AssertionError
            If the results file was not generated.
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

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file to be repaired.

        Returns
        -------
        sasa_score: float
            The SASA score of the mutated structure.
        """
        mutated_structure = parse_pdb_as_structure(
            self.working_dir / f"{pdb_file.stem}_1.pdb", structure_name="pdb_mutated"
        )

        sasa_computer = SASA.ShrakeRupley()

        # This computes the sasa score, and attaches
        # it to the structure.
        sasa_computer.compute(mutated_structure, level="S")

        return mutated_structure.sasa

    def compute_stability(self, pdb_file: Path, mutations: List[str] = None) -> float:
        """
        Compute the stability of a protein structure using FoldX.

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file of the protein structure.
        mutations : List[str], optional
            A list of mutations to be simulated. Only single mutations are supported. Pass no mutations to compute the energy of the wildtype.

        Returns
        -------
        float
            The stability of the protein structure (defined as the negative
            change in energy).

        Raises
        ------
        AssertionError
            If the number of mutations is not 0 or 1.

        """
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
        """
        Compute the solvent-accessible surface area (SASA) score for a given protein structure.

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file of the protein structure.
        mutations : List[str], optional
            A list of mutations to be simulated on the protein structure. Only single mutations are supported.
            Pass no mutations if you want to compute the SASA of the wildtype.

        Returns
        -------
        float
            The computed SASA score.

        Raises
        ------
        AssertionError
            If the number of mutations is not 0 or 1.

        """
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. Pass no mutations if you want to compute the SASA of the wildtype."
        self._simulate_mutations(pdb_file, mutations)

        sasa_score = self._compute_sasa(pdb_file)
        return sasa_score

    def compute_stability_and_sasa(self, pdb_file: Path, mutations: List[str] = None):
        """Computes stability and sasa with a single foldx run,
        instead of two separate runs.

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file of the protein structure.
        mutations : List[str], optional
            A list of mutations to be simulated on the protein structure. Only single mutations are supported.
            Pass no mutations if you want to compute the SASA of the wildtype.
        """
        if mutations is not None:
            assert len(mutations) in [
                0,
                1,
            ], "We only support single mutations for now. Pass no mutations if you want to compute the energy and SASA of the wildtype."
        self._simulate_mutations(pdb_file, mutations)

        stability = -self._read_energy(pdb_file)
        sasa_score = self._compute_sasa(pdb_file)

        return stability, sasa_score

    def copy_foldx_files(self, pdb_file: Path):
        """Copies the pdb file to the working directory.

        Parameters
        ----------
        pdb_file : Path
            The path to the PDB file of the protein structure.
        """
        if (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
            # If rotabase exists, it's likely that the user is
            # using foldx v4. We should copy it if it's not
            # already in the working directory.
            try:
                os.symlink(
                    str(PATH_TO_FOLDX_FILES / "rotabase.txt"),
                    str(self.working_dir / "rotabase.txt"),
                )
            except FileExistsError:
                pass

        destination_path_for_pdb = self.working_dir / f"{pdb_file.stem}.pdb"
        if not destination_path_for_pdb.exists():
            shutil.copy(pdb_file, destination_path_for_pdb)

    @staticmethod
    def write_mutations_to_file(
        wildtype_resiudes: List[Residue], mutations: List[str], output_dir: Path
    ) -> None:
        """Writes the list of mutations to a file
        in the given directory.

        Parameters
        ----------
        wildtype_resiudes : List[Residue]
            The list of wildtype residues.
        mutations : List[str]
            The list of mutations to simulate.
        output_dir : Path
            The directory to write the file to.
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
            mutations_in_line = mutations_from_wildtype_residues_and_mutant(
                wildtype_resiudes, mutation
            )

            # These are represented as a single line, separated
            # by commas, and terminated by a semicolon.
            lines.append(",".join(mutations_in_line) + ";")

        with open(output_dir / "individual_list.txt", "w") as f:
            f.writelines(lines)
