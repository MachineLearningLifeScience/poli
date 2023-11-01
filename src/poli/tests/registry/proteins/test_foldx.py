import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


def test_foldx_stability_is_available():
    """
    We test whether foldx_stability is available when
    1. foldx is installed.
    2. foldx files are in the right position
    2. biopython and python-levenshtein are installed
    """
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")

    from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES

    assert "foldx_stability" in AVAILABLE_PROBLEM_FACTORIES


def test_force_registering_foldx_stability():
    """
    We test whether we can force-register the foldx_stability
    problem if foldx is installed.
    """
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
        force_register=True,
    )

    assert np.isclose(y0, 32.4896).all()
    f.terminate()


def test_force_registering_foldx_sasa():
    """
    We test whether we can force-register the foldx_sasa
    problem if foldx is installed.
    """
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
        force_register=True,
    )

    assert np.isclose(y0, 8411.45578009).all()
    f.terminate()
