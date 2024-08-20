from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


@pytest.mark.poli__rasp
def test_rasp_on_3ned_against_notebooks_results():
    # If the previous import was successful, we can
    # create a RaSP problem:
    problem = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
    )
    f, x0 = problem.black_box, problem.x0

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

    assert np.isclose(y[0], 0.03654138690753095, atol=1e-4)
    assert np.isclose(y[1], -0.07091977827871465, atol=1e-4)
    assert np.isclose(y[2], -0.2835593180137258, atol=1e-4)


@pytest.mark.poli__rasp
def test_rasp_on_3ned_against_notebooks_results_isolated():
    """
    We test forceful registration of the RaSP problem.
    """
    problem = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        force_isolation=True,
    )
    f, x0 = problem.black_box, problem.x0

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


@pytest.mark.poli__rasp
def test_rasp_using_additive_flag_on_two_mutations():
    # If the previous import was successful, we can
    # create a RaSP problem:
    problem = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        additive=True,
    )
    f, x0 = problem.black_box, problem.x0

    wildtype_sequence = "".join(x0[0])
    one_mutant_with_two_mutations = [
        "AR" + wildtype_sequence[2:],
    ]
    two_mutations = [
        "A" + wildtype_sequence[1:],
        wildtype_sequence[:1] + "R" + wildtype_sequence[2:],
    ]

    x = np.array([list(mutation) for mutation in one_mutant_with_two_mutations])
    y = f(x)

    x1 = np.array([list(mutation) for mutation in two_mutations])
    y1 = f(x1)

    assert y == y1.sum()


if __name__ == "__main__":
    test_rasp_using_additive_flag_on_two_mutations()
