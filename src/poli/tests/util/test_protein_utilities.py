"""
This module tests the utilities related to proteins,
which can be found in poli.core.util.proteins.

Since this code is only intended to run inside the
poli__protein environment, we add a skip to the
whole module if the import fails.
"""

from pathlib import Path

import pytest

try:
    from poli.core.util.proteins.pdb_parsing import (
        parse_pdb_as_residue_strings,
        parse_pdb_as_residues,
    )
except ImportError:
    pytest.skip(
        "Could not import the protein utilities for parsing. ", allow_module_level=True
    )

try:
    from poli.core.util.proteins.mutations import (
        find_closest_wildtype_pdb_file_to_mutant,
        mutations_from_wildtype_residues_and_mutant,
    )
except ImportError:
    pytest.skip(
        "Could not import the protein utilities for mutations. ",
        allow_module_level=True,
    )

THIS_DIR = Path(__file__).parent.resolve()


@pytest.mark.poli__protein
class TestClosestPDBFilesToMutation:
    wildtype_pdb_paths = [
        THIS_DIR / "3ned.pdb",
        THIS_DIR / "2vae.pdb",
    ]

    wildtype_strings = [
        parse_pdb_as_residue_strings(pdb_path) for pdb_path in wildtype_pdb_paths
    ]

    wildtype_residues = [
        parse_pdb_as_residues(pdb_path) for pdb_path in wildtype_pdb_paths
    ]

    one_mutation_to_3ned = "".join(wildtype_strings[0])
    one_mutation_to_3ned = "A" + one_mutation_to_3ned[1:]

    def test_finding_closest_pdb_to_mutation(self):
        """
        Loads up two PDB files, computes a couple of mutations
        of one, and then finds the closest PDB file to the
        mutated one.
        """
        # We check whether the closest PDB file to the mutated
        # one is 3ned.
        closest_pdb_path = find_closest_wildtype_pdb_file_to_mutant(
            wildtype_pdb_files=self.wildtype_pdb_paths,
            mutated_residue_string=self.one_mutation_to_3ned,
        )
        assert closest_pdb_path == self.wildtype_pdb_paths[0]

    def test_finding_closest_pdb_is_agnostic_to_lowercase(self):
        """
        Loads up two PDB files, computes a couple of mutations
        of one, and then finds the closest PDB file to the
        mutated one.
        """
        # We check whether the closest PDB file to the mutated
        # one is 3ned, but now we use lowercase.
        closest_pdb_path = find_closest_wildtype_pdb_file_to_mutant(
            wildtype_pdb_files=self.wildtype_pdb_paths,
            mutated_residue_string=self.one_mutation_to_3ned.lower(),
        )
        assert closest_pdb_path == self.wildtype_pdb_paths[0]

    def test_computing_mutations_from_wildtype_and_mutant(self):
        """
        We test whether we can compute the mutations from
        a wildtype and a mutant.
        """
        mutations = mutations_from_wildtype_residues_and_mutant(
            wildtype_residues=self.wildtype_residues[0],
            mutated_residue_string=self.one_mutation_to_3ned,
        )

        assert mutations == ["EA1A"]

    def test_computing_mutations_is_agnostic_to_lowercase(self):
        """
        We test whether we can compute the mutations from
        a wildtype and a mutant.
        """
        mutations = mutations_from_wildtype_residues_and_mutant(
            wildtype_residues=self.wildtype_residues[0],
            mutated_residue_string=self.one_mutation_to_3ned.lower(),
        )

        assert mutations == ["EA1A"]
