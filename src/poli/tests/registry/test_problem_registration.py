import pytest
from pathlib import Path

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
    _, f, x0, y0, _ = objective_factory.create(name="aloha")
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

    print(AVAILABLE_PROBLEM_FACTORIES)
    assert "rdkit_qed" in AVAILABLE_PROBLEM_FACTORIES


def test_logp_is_available():
    """
    We test whether the logp problem is available
    when rdkit and selfies are installed.
    """
    _ = pytest.importorskip("rdkit")
    _ = pytest.importorskip("selfies")
    from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES

    print(AVAILABLE_PROBLEM_FACTORIES)
    assert "rdkit_logp" in AVAILABLE_PROBLEM_FACTORIES


def test_force_registering_qed():
    """
    We test whether we can force-register the qed problem
    if rdkit and selfies are not installed.
    """
    _, f, x0, y0, _ = objective_factory.create(
        name="rdkit_qed",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
        force_register=True,
    )
    f.terminate()


def test_force_registering_logp():
    """
    We test whether we can force-register the logp problem
    if rdkit and selfies are not installed.
    """
    _, f, x0, y0, _ = objective_factory.create(
        name="rdkit_logp",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
        force_register=True,
    )
    f.terminate()


# def test_force_registering_smb():
#     # assert False
#     print("Testing SMB")
#     _, f, x0, y0, _ = objective_factory.create(
#         name="super_mario_bros",
#         force_register=True,
#     )
#     f.terminate()


def test_registering_qed():
    # assert False
    print("Testing QED")
    """
    Testing whether we can register the qed problem
    if rdkit and selfies are installed.
    """
    rdkit = pytest.importorskip("rdkit")
    selfies = pytest.importorskip("selfies")
    np = pytest.importorskip("numpy")

    _, f, x0, y0, _ = objective_factory.create(
        name="rdkit_qed",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
    )
    x = np.array([[1]])
    f(x)
    f.terminate()


def test_registering_logp():
    # assert False
    print("Testing LogP")
    """
    Testing whether we can register the logp problem
    if rdkit and selfies are installed.
    """
    THIS_DIR = Path(__file__).parent.resolve()
    rdkit = pytest.importorskip("rdkit")
    selfies = pytest.importorskip("selfies")
    np = pytest.importorskip("numpy")

    _, f, x0, y0, _ = objective_factory.create(
        name="rdkit_logp",
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
    )
    x = np.array([[1]])
    f(x)
    f.terminate()
