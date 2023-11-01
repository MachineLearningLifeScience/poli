import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


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
