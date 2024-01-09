import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

HOME_DIR = Path().home().resolve()
PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
if not PATH_TO_FOLDX_FILES.exists():
    pytest.skip("FoldX is not installed. ", allow_module_level=True)

if not (PATH_TO_FOLDX_FILES / "foldx").exists():
    pytest.skip("FoldX is not compiled. ", allow_module_level=True)


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

    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")
    _ = pytest.importorskip("pdbtools")

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

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
        force_register=True,
    )

    assert np.isclose(y0, 8411.45578009).all()
    f.terminate()


def test_registering_foldx_stability():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    if not (THIS_DIR / "101m_Repair.pdb").exists():
        pytest.skip("Could not find wildtype 101m_Repair.pdb in test folder.")

    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )

    assert np.isclose(y0, 32.4896).all()


def test_registering_foldx_sasa():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )

    assert np.isclose(y0, 8411.45578009).all()


def test_registering_foldx_stability_and_sasa():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )

    assert np.isclose(y0[:, 0], 32.4896).all()
    assert np.isclose(y0[:, 1], 8411.45578009).all()


@pytest.mark.slow()
def test_foldx_from_non_repaired_file():
    """
    In this test, we check whether foldx properly
    repairs a file if it doesn't contain _Repair.

    TODO: mock the behavior of the repair function
    inside the foldx interface. Otherwise, this test
    takes 4min to run.
    """
    wildtype_pdb_path = THIS_DIR / "3ned.pdb"
    _, f, _, y0, _ = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=wildtype_pdb_path,
        eager_repair=True,
    )

    assert np.isclose(y0, 32.6135).all()


def test_foldx_from_repaired_file():
    """
    In this test, we check whether no repair is
    performed if the file already contains _Repair.
    """
    wildtype_pdb_path = THIS_DIR / "101m_Repair.pdb"
    _, f, _, y0, _ = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=wildtype_pdb_path,
    )

    assert np.isclose(y0, 32.4896).all()


if __name__ == "__main__":
    # test_foldx_stability_is_available()
    test_force_registering_foldx_sasa()
