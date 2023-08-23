import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


def test_registering_white_noise():
    print("testing white noise")
    np = pytest.importorskip("numpy")
    _, f, _, _, _ = objective_factory.create(name="white_noise")
    x = np.array([["A", "B", "C", "D"]])
    f(x)
    f.terminate()


def test_registering_aloha():
    np = pytest.importorskip("numpy")
    _, f, _, y0, _ = objective_factory.create(name="aloha")
    x = np.array([list("ALOOF")])
    assert f(x) == 3
    f.terminate()


def test_qed_is_available():
    """
    We test whether the qed problem is available
    when rdkit and selfies are installed.
    """
    _ = pytest.importorskip("rdkit")
    _ = pytest.importorskip("selfies")
    from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES

    assert "rdkit_qed" in AVAILABLE_PROBLEM_FACTORIES


def test_logp_is_available():
    """
    We test whether the logp problem is available
    when rdkit and selfies are installed.
    """
    _ = pytest.importorskip("rdkit")
    _ = pytest.importorskip("selfies")
    from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES

    assert "rdkit_logp" in AVAILABLE_PROBLEM_FACTORIES


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


def test_force_registering_qed():
    """
    We test whether we can force-register the qed problem
    if rdkit and selfies are not installed.
    """
    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_qed",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
        force_register=True,
    )

    # Asserting that the QED of a single carbon
    # is close to 0.35978494 (according to RDKit).
    assert np.isclose(y0, 0.35978494).all()
    f.terminate()


def test_force_registering_logp():
    """
    We test whether we can force-register the logp problem
    if rdkit and selfies are not installed.
    """
    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_logp",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
        force_register=True,
    )

    # Asserting that a single carbon atom has logp close
    # to 0.6361. (according to RDKit)
    assert np.isclose(y0, 0.6361).all()
    f.terminate()


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


# TODO: Automated testing via GitHub actions
# struggles with this test, even after adding
# xvbf coactions. We need to add/raise errors
# in the objective function itself to see
# why that is...

# For now, we'll remove it.

# def test_force_registering_smb():
#     # assert False
#     print("Testing SMB")
#     _, f, _, y0, _ = objective_factory.create(
#         name="super_mario_bros",
#         force_register=True,
#     )
#     f.terminate()


def test_registering_qed():
    """
    Testing whether we can register the qed problem
    if rdkit and selfies are installed.
    """
    _ = pytest.importorskip("rdkit")
    _ = pytest.importorskip("selfies")
    np = pytest.importorskip("numpy")

    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_qed",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
    )
    x = np.array([[1]])
    f(x)

    # Asserting that the QED of a single carbon
    # is close to 0.35978494 (according to RDKit).
    assert np.isclose(y0, 0.35978494).all()

    f.terminate()


def test_registering_logp():
    """
    Testing whether we can register the logp problem
    if rdkit and selfies are installed.
    """
    rdkit = pytest.importorskip("rdkit")
    selfies = pytest.importorskip("selfies")
    np = pytest.importorskip("numpy")

    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_logp",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
    )
    x = np.array([[1]])
    f(x)

    # Asserting that a single carbon atom has logp close
    # to 0.6361. (according to RDKit)
    assert np.isclose(y0, 0.6361).all()

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

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

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

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

    _ = pytest.importorskip("Bio")
    _ = pytest.importorskip("Levenshtein")

    _, f, _, y0, _ = objective_factory.create(
        name="foldx_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )

    assert np.isclose(y0, 8411.45578009).all()
