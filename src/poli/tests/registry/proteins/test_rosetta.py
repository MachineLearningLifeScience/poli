from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

@pytest.mark.poli__rosetta_energy
@pytest.mark.parametrize("unit", ["DDG", "REU", "DREU"])
def test_rosetta_wt_zero_ddg(unit):
    """
    Test that the WT score is equal to x0 evaluated on f
    """
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "1ggx.pdb",
        relax=False,  # fast compute
        pack=False,
        unit = unit
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    assert f.wt_score == y0


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_against_known_results():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
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

    # Asserting that the results are the consistent:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    assert np.isclose(y[0], -0.03654138690753095, atol=1e-4)
    assert np.isclose(y[1], 0.07091977827871465, atol=1e-4)
    assert np.isclose(y[2], 0.2835593180137258, atol=1e-4)


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_evaluating_twice_vs_once():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
    )
    f, x0 = problem.black_box, problem.x0

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
        "N" + wildtype_sequence[1:],
    ]

    two_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
    ]
    one_mutation = [
        "N" + wildtype_sequence[1:],
    ]

    x_all = np.array([list(mutation) for mutation in three_mutations])
    y = f(x_all)

    y_two = f(np.array([list(mutation) for mutation in two_mutations]))
    y_one = f(np.array([list(mutation) for mutation in one_mutation]))

    assert y[0] == y_two[0]
    assert y[1] == y_two[1]
    assert y[2] == y_one[0]
    # Asserting that the results are the same as in the
    # notebook:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    assert np.isclose(y[0], -0.03654138690753095, atol=1e-4)
    assert np.isclose(y[1], 0.07091977827871465, atol=1e-4)
    assert np.isclose(y[2], 0.2835593180137258, atol=1e-4)


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_against_notebooks_results_isolated():
    """
    We test forceful registration of the Rosetta problem.
    """
    problem = objective_factory.create(
        name="rosetta_energy",
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
    assert np.isclose(y[0], -0.0365, atol=1e-4)
    assert np.isclose(y[1], 0.07091, atol=1e-4)
    assert np.isclose(y[2], 0.283559, atol=1e-4)


@pytest.mark.poli__rosetta_energy
def test_rosetta_fails_on_invalid_amino_acids():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
    )
    f, x0 = problem.black_box, problem.x0

    # This is an unfeasible mutation, since joining
    # all the strings would result in a sequence
    # that is _not_ the same length as the wildtype.
    problematic_x = x0.copy()
    problematic_x[0][0] = "B"
    with pytest.raises(ValueError):
        f(problematic_x)


@pytest.mark.poli__rosetta_energy
@pytest.mark.slow
def test_rosetta_relax_pack():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=True,  # fast compute
        pack=True,
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

    # Asserting that the results are the consistent:
    # E1A: 0.03654138690753095
    # E1R: -0.07091977827871465
    # E1N: -0.2835593180137258

    assert np.isclose(y[0], -0.03654138690753095, atol=1e-4)
    assert np.isclose(y[1], 0.07091977827871465, atol=1e-4)
    assert np.isclose(y[2], 0.2835593180137258, atol=1e-4)
