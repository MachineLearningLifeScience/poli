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
    alphabet = ["", "C", "..."]
    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_qed",
        alphabet=alphabet,
        force_register=True,
    )

    # Asserting that the QED of a single carbon
    # is close to 0.35978494 (according to RDKit).
    assert np.isclose(y0, 0.35978494).all()
    f.terminate()


def test_force_registering_qed_with_context_manager():
    """
    Tests the objective_factory.start method on QED.
    """
    with objective_factory.start(
        name="rdkit_qed",
        force_register=True,
        force_isolation=True,
        path_to_alphabet=THIS_DIR / "alphabet_qed.json",
    ) as f:
        x = np.array([["C"]])
        y = f(x)
        assert np.isclose(y, 0.35978494).all()


def test_force_registering_logp():
    """
    We test whether we can force-register the logp problem
    if rdkit and selfies are not installed.
    """
    alphabet = ["", "C", ""]
    _, f, _, y0, _ = objective_factory.create(
        name="rdkit_logp",
        alphabet=alphabet,
        # path_to_alphabet=THIS_DIR / "alphabet_qed.json",
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
    x = np.array([["C"]])
    y = f(x)

    # Asserting that the QED of a single carbon
    # is close to 0.35978494 (according to RDKit).
    assert np.isclose(y, 0.35978494).all()

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

    if not (PATH_TO_FOLDX_FILES / "rotabase.txt").exists():
        pytest.skip("rotabase.txt is not in the foldx directory. ")

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
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )

    assert np.isclose(y0[:, 0], 32.4896).all()
    assert np.isclose(y0[:, 1], 8411.45578009).all()


def test_penalized_logp_lambo():
    """
    Testing whether we can register the logp problem
    from lambo.
    """
    from poli import objective_factory

    _ = pytest.importorskip("lambo")

    # Using create
    _, f, x0, y0, _ = objective_factory.create(
        name="penalized_logp_lambo", force_register=True
    )
    print(x0)
    print(y0)
    f.terminate()


def test_rasp_on_3ned_against_notebooks_results_on_rasp_env():
    try:
        from poli.objective_repository.rasp.register import RaspProblemFactory
    except ImportError:
        pytest.skip("Could not import RaspProblemFactory. ")

    import torch

    # For us to match what the notebook says, we have
    # to run at double precision.
    torch.set_default_dtype(torch.float64)

    # If the previous import was successful, we can
    # create a RaSP problem:
    _, f, x0, _, _ = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
    )

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
        "N" + wildtype_sequence[1:],
    ]

    x = np.array([list(mutation) for mutation in three_mutations])
    y = f(x)

    # Asserting that the results are the same as in the
    # notebook:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    assert np.isclose(y[0], 0.03654138690753095)
    assert np.isclose(y[1], -0.07091977827871465)
    assert np.isclose(y[2], -0.2835593180137258)


def test_rasp_on_3ned_against_notebooks_results_isolated():
    """
    We test forceful registration of the RaSP problem.
    """
    # If the previous import was successful, we can
    # create a RaSP problem:
    _, f, x0, _, _ = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
    )

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
        "N" + wildtype_sequence[1:],
    ]

    x = np.array([list(mutation) for mutation in three_mutations])
    y = f(x)

    # Asserting that the results are the same as in the
    # notebook:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    # Notice how we are clipping the actual values,
    # this is because we would need double precision
    # to test against the exact values described above.
    # TODO: Should we do double precision by default
    # inside the RaSP problem?
    assert np.isclose(y[0], 0.0365, atol=1e-4)
    assert np.isclose(y[1], -0.07091, atol=1e-4)
    assert np.isclose(y[2], -0.283559, atol=1e-4)


def test_rasp_on_3ned_against_notebooks_results_on_rasp_env():
    try:
        from poli.objective_repository.rasp.register import RaspProblemFactory
    except ImportError:
        pytest.skip("Could not import RaspProblemFactory. ")

    import torch

    # For us to match what the notebook says, we have
    # to run at double precision.
    torch.set_default_dtype(torch.float64)

    # If the previous import was successful, we can
    # create a RaSP problem:
    _, f, x0, _, _ = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
    )

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
        "N" + wildtype_sequence[1:],
    ]

    x = np.array([list(mutation) for mutation in three_mutations])
    y = f(x)

    # Asserting that the results are the same as in the
    # notebook:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    assert np.isclose(y[0], 0.03654138690753095)
    assert np.isclose(y[1], -0.07091977827871465)
    assert np.isclose(y[2], -0.2835593180137258)


if __name__ == "__main__":
    test_rasp_on_3ned_against_notebooks_results_isolated()
