"""
This module tests the foldx interface, which can be found in
poli.core.util.proteins.foldx

Since this code is only intended to run inside the
poli__protein environment, we add a skip to the
whole module if the import fails.
"""

from pathlib import Path

import pytest

try:
    from poli.core.util.proteins.foldx import FoldxInterface
except (ImportError, FileNotFoundError):
    pytest.skip("Could not import the foldx interface. ", allow_module_level=True)

try:
    from poli.core.util.proteins.pdb_parsing import (  # noqa F401
        parse_pdb_as_residue_strings,
    )
except ImportError:
    pytest.skip(
        "Could not import the protein utilities for parsing. ", allow_module_level=True
    )

THIS_DIR = Path(__file__).parent.resolve()


@pytest.mark.poli__protein
class TestFoldxInterface:
    wildtype_pdb_path = THIS_DIR / "3ned.pdb"
    tmp_path = THIS_DIR / "tmp"

    foldx_interface = FoldxInterface(tmp_path)

    def test_copying_files(self):
        """
        Tests that the files are copied correctly.
        """
        self.tmp_path.mkdir(exist_ok=True)

        self.foldx_interface.copy_foldx_files(self.wildtype_pdb_path)

        assert (self.foldx_interface.working_dir / "rotabase.txt").exists()
        assert (
            self.foldx_interface.working_dir / f"{self.wildtype_pdb_path.stem}.pdb"
        ).exists()

        # Clean up
        (self.foldx_interface.working_dir / "rotabase.txt").unlink()
        (
            self.foldx_interface.working_dir / f"{self.wildtype_pdb_path.stem}.pdb"
        ).unlink()

    def test_specifying_string_working_dir(self):
        new_foldx_interface = FoldxInterface(str(self.tmp_path))

        new_foldx_interface.copy_foldx_files(self.wildtype_pdb_path)

        assert (new_foldx_interface.working_dir / "rotabase.txt").exists()
        assert (
            new_foldx_interface.working_dir / f"{self.wildtype_pdb_path.stem}.pdb"
        ).exists()

        # Clean up
        (new_foldx_interface.working_dir / "rotabase.txt").unlink()
        (
            new_foldx_interface.working_dir / f"{self.wildtype_pdb_path.stem}.pdb"
        ).unlink()
