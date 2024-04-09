import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory
from poli.objective_repository import GSK3BetaBlackBox, DRD2BlackBox, JNK3BlackBox

THIS_DIR = Path(__file__).parent.resolve()

SEED = np.random.randint(0, 1000)


def test_force_registering_qed():
    """
    We test whether we can force-register the qed problem
    if rdkit and selfies are not installed.
    """
    problem = objective_factory.create(
        name="rdkit_qed",
        force_register=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

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
    ) as f:
        x = np.array([["C"]])
        y = f(x)
        assert np.isclose(y, 0.35978494).all()


def test_force_registering_logp():
    """
    We test whether we can force-register the logp problem
    if rdkit and selfies are not installed.
    """
    problem = objective_factory.create(
        name="rdkit_logp",
        force_register=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

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
    problem = objective_factory.create(name="penalized_logp_lambo", force_register=True)


def test_querying_dockstring_using_smiles():
    """
    In this test, we force-register and query dockstring.
    """
    from poli import objective_factory

    problem = objective_factory.create(
        name="dockstring",
        target_name="DRD2",
        string_representation="SMILES",
        force_register=True,
    )
    f = problem.black_box

    # Docking another smiles
    x1 = np.array([list("CC(=O)OC1=CC=CC=C1C(=O)O")])
    y1 = f(x1)

    f.terminate()


def test_querying_dockstring_using_selfies():
    """
    In this test, we check whether dockstring still
    works when using SELFIES instead of SMILES.
    """
    from poli import objective_factory

    problem = objective_factory.create(
        name="dockstring",
        target_name="ABL1",
        string_representation="SELFIES",
        force_register=True,
    )
    f = problem.black_box

    # Docking another smiles
    selfies_aspirin = np.array(
        [
            [
                "[C]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[Ring1]",
                "[=Branch1]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
            ]
        ]
    )

    y1 = f(selfies_aspirin)
    f.terminate()


test_data_for_pmo = [
    # Removed until TDC fixes issue #238.
    # ("gsk3_beta", GSK3BetaBlackBox, {"string_representation": "SMILES"}, 0.03)
    ("gsk3_beta", GSK3BetaBlackBox, {"string_representation": "SMILES"}, None),
    (
        "drd2_docking",
        DRD2BlackBox,
        {"string_representation": "SMILES"},
        0.0015465365340340924,
    ),
    # Removed until TDC fixes issue #238.
    # ("jnk3", JNK3BlackBox, {"string_representation": "SMILES"}, 0.01),
    # ("jnk3", JNK3BlackBox, {"string_representation": "SMILES"}, None),
]


@pytest.mark.parametrize(
    "black_box_name, black_box_class, kwargs_for_black_box, value_to_check",
    test_data_for_pmo,
)
def test_pmo_black_boxes(
    black_box_name, black_box_class, kwargs_for_black_box, value_to_check
):
    from poli import create
    from poli.core.util.seeding import seed_python_numpy_and_torch

    problem = create(
        name=black_box_name,
        seed=SEED,
        **kwargs_for_black_box,
    )
    x0 = problem.x0
    y0 = problem.black_box(x0)

    seed_python_numpy_and_torch(seed=SEED)
    f = black_box_class(**kwargs_for_black_box)
    y0_ = f(x0)

    assert np.allclose(y0_, y0)
    if value_to_check is not None:
        assert (y0_ == value_to_check).all()


if __name__ == "__main__":
    test_pmo_black_boxes(*test_data_for_pmo[-1])
