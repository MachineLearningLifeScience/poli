"""
This module contains utilities for defining
mutations on proteins according to foldx.

foldx expects mutations in a certain format:
    - the first letter is the original residue
    - the second letter is the position of the mutation
    - the third letter is the chain ID,
    - the fourth letter is the mutant residue.
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
    Overwriting editops to only consider replacements between strings.
    This returns ("replace", pos_in_string_1, pos_in_string_2).
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
    """
    Since foldx expects an individual_list.txt file of mutations,
    this function computes the Levenshtein distance between
    the wildtype residue string and the mutated residue string,
    keeping track of the replacements.

    This method returns a list of strings which are to be written
    in a single line of individual_list.txt.

    If the mutated residue string is the same as the wildtype residue
    string, we still need to pass a dummy mutation to foldx, so we
    "mutate" the first residue in the wildtype string to itself.

    TODO: add description of inputs and outputs
    """
    wildtype_residue_string = "".join(
        [seq1(res.get_resname()) for res in wildtype_residues]
    )

    assert len(mutated_residue_string) == len(wildtype_residue_string), (
        f"wildtype residue string and mutated residue string "
        f"have different lengths: {len(wildtype_residue_string)} "
        f"and {len(mutated_residue_string)}."
    )

    # If the mutated string is the same as the wildtype string,
    # there are no mutations. Still, FoldX expects us to pass
    # a dummy mutation, so we "mutate" the first residue in the
    # wildtype string.
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
                wildtype_residue_string[i] != mutated_residue_string[i]
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
