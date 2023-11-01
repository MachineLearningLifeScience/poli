import pytest
from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


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
        force_register=True,
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