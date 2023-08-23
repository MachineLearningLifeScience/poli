"""
This module contains utilities for
loading PDB files and parsing them.
"""

from pathlib import Path
from typing import Tuple, List

from Bio import PDB
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1


def parse_pdb_as_structure(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> PDB.Structure.Structure:
    """
    Parses the PDB file at the given path and returns
    the structure. The parsing is done quietly by default,
    but you can set verbose=True to get some output.
    """
    parser = PDB.PDBParser(QUIET=not verbose)
    return parser.get_structure(structure_name, path_to_pdb)


def parse_pdb_as_residues(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> List[Residue]:
    """
    TODO: document
    """
    structure = parse_pdb_as_structure(path_to_pdb, structure_name, verbose)
    return list(structure.get_residues())


def parse_pdb_as_residue_strings(
    path_to_pdb: Path, structure_name: str = "pdb", verbose: bool = False
) -> List[str]:
    """
    TODO: document
    """
    residues = parse_pdb_as_residues(path_to_pdb, structure_name, verbose)

    return [seq1(res.get_resname()) for res in residues]
