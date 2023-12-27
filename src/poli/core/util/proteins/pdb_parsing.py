"""This module contains utilities for loading PDB files and parsing them.
"""

from pathlib import Path
from typing import List

from Bio import PDB
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1


def parse_pdb_as_structure(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> PDB.Structure.Structure:
    """Parses the PDB file at the given path and returns
    the structure.

    The parsing is done quietly by default,
    but you can set verbose=True to get some output.

    Parameters
    -----------
    path_to_pdb : Path
        The path to the PDB file.
    structure_name : str, optional
        The name of the structure (which is passed to the get_structure
        method of the PDBParser object). Defaults to "pdb".
    verbose : bool, optional
        If True, print the progress of the parsing. Defaults to False.

    Returns
    --------
    PDB.Structure.Structure
        The parsed structure.
    """
    parser = PDB.PDBParser(QUIET=not verbose)
    return parser.get_structure(structure_name, path_to_pdb)


def parse_pdb_as_residues(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> List[Residue]:
    """
    Parse a PDB file and return a list of Residue objects.

    Parameters
    -----------
    path_to_pdb: Path
        The path to the PDB file.
    structure_name: str, optional
        The name of the structure. Defaults to "pdb".
    verbose: bool, optional
        Whether to print verbose output. Defaults to False.

    Returns
    --------
        residues: List[Residue]
            A list of Residue objects representing the parsed PDB file.
    """
    structure = parse_pdb_as_structure(path_to_pdb, structure_name, verbose)
    return list(structure.get_residues())


def parse_pdb_as_residue_strings(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> List[str]:
    """
    Parse a PDB file and return a list of residue strings.

    Parameters
    ----------
    path_to_pdb : Path
        The path to the PDB file.
    structure_name : str, optional
        The name of the structure, by default "pdb".
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    List[str]
        A list of residue strings.
    """
    residues = parse_pdb_as_residues(path_to_pdb, structure_name, verbose)

    return [seq1(res.get_resname()) for res in residues]
