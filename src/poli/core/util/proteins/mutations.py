"""Utilities for defining and manipulating mutations on proteins.

This module contains utilities for defining
mutations on proteins according to foldx.

foldx expects mutations in a certain format:
    - the first letter is the original residue
    - the second letter is the position of the mutation
    - the third letter is the chain ID,
    - the fourth letter is the mutant residue.

See the "Individual List Mode" of https://foldxsuite.crg.eu/parameter/mutant-file
for more details.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1

from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residue_strings


def edits_between_strings(
    string_1: str, string_2: str, strict: bool = True
) -> List[Tuple[str, int, int]]:
    """
    Compute the edit operations between two strings.

    Parameters
    ----------
    string_1 : str
        The first string.
    string_2 : str
        The second string.
    strict : bool, optional
        If True, check if the lengths of string_1 and string_2 are equal.
        Defaults to True.

    Returns
    -------
    List[Tuple[str, int, int]]
        A list of tuples representing the edit operations between the two strings.
        Each tuple contains the operation type ("replace"), the position in string_1,
        and the position in string_2.

    Raises
    ------
    AssertionError
        If strict is True and the lengths of string_1 and string_2 are different.

    Examples
    --------
    >>> edits_between_strings("abc", "abd")
    [('replace', 2, 2)]

    >>> edits_between_strings("abc", "def")
    [('replace', 0, 0), ('replace', 1, 1), ('replace', 2, 2)]
    """
    if strict:
        assert len(string_1) == len(string_2), (
            f"string_1 and string_2 have different lengths: "
            f"{len(string_1)} and {len(string_2)}."
        )
    for i, (a, b) in enumerate(zip(string_1, string_2)):
        if a != b:
            yield ("replace", i, i)


def mutations_from_wildtype_residues_and_mutant(
    wildtype_residues: List[Residue], mutated_residue_string: str
) -> List[str]:
    """Computes the mutations from a wildtype list of residues
    and a mutated residue string.

    Since foldx expects an individual_list.txt file of mutations,
    this function computes the Levenshtein distance between
    the wildtype residue string and the mutated residue string,
    keeping track of the replacements.

    This method returns a list of strings which are to be written
    in a single line of individual_list.txt. Each string is a
    mutation in the format foldx expects (e.g. "EA1R", meaning
    that an E was mutated to an R in position 1 of chain A. The
    first letter is the original residue, the second letter is
    the chain, the third letter is the position, and the fourth
    letter is the mutant residue).

    If the mutated residue string is the same as the wildtype residue
    string, we still need to pass a dummy mutation to foldx, so we
    "mutate" the first residue in the wildtype string to itself.

    For example:
        wildtype_residue_string = "ECDE..."
        mutated_residue_string =  "ACDE..."

    This function would return (assuming that we are mutating the
    chain "A"):
        ["EA1A"]

    Parameters
    ----------
    wildtype_residues : List[Residue]
        The list of wildtype residues.
    mutated_residue_string : str
        The mutated residue string.

    Returns
    -------
    mutations: List[str]
        The list of mutations in the format foldx expects.
    """
    wildtype_residue_string = "".join(
        [seq1(res.get_resname()) for res in wildtype_residues]
    )

    # Making sure we treat the mutant string as uppercase
    mutated_residue_string = mutated_residue_string.upper()

    assert len(mutated_residue_string) == len(wildtype_residue_string), (
        f"wildtype residue string and mutated residue string "
        f"have different lengths: {len(wildtype_residue_string)} "
        f"and {len(mutated_residue_string)}."
    )

    # If the mutated string is the same as the wildtype string,
    # there are no mutations. Still, FoldX expects us to pass
    # a dummy mutation, so we "mutate" the first residue in the
    # wildtype string to itself.
    if mutated_residue_string == wildtype_residue_string:
        first_residue = wildtype_residues[0]
        first_residue_name = seq1(first_residue.get_resname())
        chain_id = first_residue.get_parent().id
        index_in_sequence = first_residue.id[1]
        return [
            f"{first_residue_name}{chain_id}{index_in_sequence}{first_residue_name}"
        ]

    mutations = edits_between_strings(wildtype_residue_string, mutated_residue_string)

    mutations_in_line = []
    for mutation_type, pos_in_wildtype, pos_in_mutant in mutations:
        assert mutation_type == "replace"
        original_wildtype_residue = wildtype_residue_string[pos_in_wildtype]
        chain_id = wildtype_residues[pos_in_wildtype].get_parent().id
        index_in_sequence = wildtype_residues[pos_in_wildtype].id[1]
        mutated_residue = mutated_residue_string[pos_in_mutant]

        mutation_string = (
            f"{original_wildtype_residue}{chain_id}{index_in_sequence}{mutated_residue}"
        )
        mutations_in_line.append(mutation_string)

    return mutations_in_line


def find_closest_wildtype_pdb_file_to_mutant(
    wildtype_pdb_files: List[Path],
    mutated_residue_string: str,
    return_hamming_distance: bool = False,
) -> Union[Path, Tuple[Path, int]]:
    """
    Find the closest wildtype PDB file to a given mutant residue string.

    Parameters
    ----------
    wildtype_pdb_files : List[Path]
        A list of paths to wildtype PDB files.
    mutated_residue_string : str
        The mutated residue string.
    return_hamming_distance : bool, optional
        If True, return the hamming distance along with the best candidate PDB file.
        Default is False.

    Returns
    -------
    Union[Path, Tuple[Path, int]]
        If return_hamming_distance is True, returns a tuple containing the best candidate PDB file
        and the hamming distance. Otherwise, returns the best candidate PDB file.

    Raises
    ------
    ValueError
        If no PDB file of the same length as the mutated residue string is found.

    """
    # First, we load up these pdb files as residue strings
    wildtype_residue_strings = {
        pdb_file: "".join(parse_pdb_as_residue_strings(pdb_file))
        for pdb_file in wildtype_pdb_files
    }

    # Since foldx only allows for substitutions, we only
    # consider the PDBs whose length is the same as the
    # mutated residue string.
    best_candidate_pdb_file = None
    min_hamming_distance = np.inf
    for pdb_file, wildtype_residue_string in wildtype_residue_strings.items():
        if len(wildtype_residue_string) != len(mutated_residue_string):
            continue

        hamming_distance = np.sum(
            [
                wildtype_residue_string.upper()[i] != mutated_residue_string.upper()[i]
                for i in range(len(wildtype_residue_string))
            ]
        )

        if hamming_distance < min_hamming_distance:
            best_candidate_pdb_file = pdb_file
            min_hamming_distance = hamming_distance

    if best_candidate_pdb_file is None:
        raise ValueError(
            f"Could not find a PDB file of length {len(mutated_residue_string)}." + "\n"
            "Are you sure you provided a valid mutant string?\n"
            f"Lengths allowed: {set([len(x) for x in wildtype_residue_strings.values()])}"
        )

    if return_hamming_distance:
        return best_candidate_pdb_file, min_hamming_distance
    else:
        return best_candidate_pdb_file
