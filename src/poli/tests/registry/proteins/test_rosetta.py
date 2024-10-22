import io
import os
from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


@pytest.fixture(scope="session", autouse=True)
def cleanup_optin_file():
    file_path = (
        THIS_DIR.parent.parent.parent
        / "objective_repository"
        / "rosetta_energy"
        / ".pyrosetta_accept.txt"
    )
    cleaned_pdbs = THIS_DIR.glob("*.clean.pdb")  # created by Rosetta during runtime

    # for individual tests, the opt-in file is required for use...
    if not file_path.exists():
        with open(file_path, "w") as file:
            file.write("accepted")

    yield  # control to the tests

    if file_path.exists():
        os.remove(file_path)
    for file in cleaned_pdbs:  # cleanup created PDBs
        os.remove(file)


@pytest.mark.poli__rosetta_energy
def test_rosetta_optin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Y"))
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
    )
    assert problem is not None


@pytest.mark.poli__rosetta_energy
@pytest.mark.parametrize("unit", ["DDG", "REU", "DREU"])
def test_rosetta_wt_zero_ddg(unit):
    """
    Test that the WT score is equal to x0 evaluated on f
    """
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
        unit=unit,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    if unit == "REU":
        assert f.inner_function.wt_score == y0
    else:
        assert np.isclose(y0, 0.0)


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_sequence_mutations_correct():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=False,  # fast compute
        pack=False,
        clean=False,  # keep sequences as loaded for consistent testing against PDB
    )
    f, x0 = problem.black_box, problem.x0

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        wildtype_sequence[:4] + "R" + wildtype_sequence[5:],
        wildtype_sequence[:9] + "N" + wildtype_sequence[10:],
    ]

    x = np.array([list(mutation) for mutation in three_mutations])
    f(x)  # function call required to populate x_t property

    # Asserting that the mutations are according to expectations
    # E1A
    # M5R
    # E10N

    for i, mutant in enumerate(three_mutations):
        assert mutant[:20] == f.inner_function.x_t[i][:20]


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
    # E1A: -3.85851953
    # E1R: -3.6878339
    # E1N: -3.80845599

    assert np.isclose(y[0], -3.85851953, atol=1e-4)
    assert np.isclose(y[1], -3.6878339, atol=1e-4)
    assert np.isclose(y[2], -3.80845599, atol=1e-4)


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_cartesian_against_known_results():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        score_function="ref2015_cart",
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
    # E1A: -4.06154293
    # E1R: -3.95461636
    # E1N: -4.08711725

    assert np.isclose(y[0], -4.06154293, atol=1e-4)
    assert np.isclose(y[1], -3.95461636, atol=1e-4)
    assert np.isclose(y[2], -4.08711725, atol=1e-4)


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
    # E1A: -3.85851953
    # E1R: -3.6878339
    # E1N: -3.80845599

    assert np.isclose(y[0], -3.85851953, atol=1e-4)
    assert np.isclose(y[1], -3.6878339, atol=1e-4)
    assert np.isclose(y[2], -3.80845599, atol=1e-4)


@pytest.mark.poli__rosetta_energy
def test_rosetta_on_3ned_against_results_isolated():
    """
    We test forceful registration of the Rosetta problem.
    """
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        force_isolation=True,
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

    # Asserting that the results are the same as in the
    # notebook:
    # E1A: -3.85851953
    # E1R: -3.6878339
    # E1N: -3.80845599

    assert np.isclose(y[0], -3.85851953, atol=1e-4)
    assert np.isclose(y[1], -3.6878339, atol=1e-4)
    assert np.isclose(y[2], -3.80845599, atol=1e-4)


@pytest.mark.poli__rosetta_energy
@pytest.mark.slow
def test_rosetta_relax_pack():
    problem = objective_factory.create(
        name="rosetta_energy",
        wildtype_pdb_path=THIS_DIR / "3ned.pdb",
        relax=True,  # extended protocol compute
        pack=True,
        seed=0,
    )
    f, x0 = problem.black_box, problem.x0

    wildtype_sequence = "".join(x0[0])
    three_mutations = [
        "A" + wildtype_sequence[1:],
        "R" + wildtype_sequence[1:],
    ]

    x = np.array([list(mutation) for mutation in three_mutations])
    y = f(x)

    # Asserting that the results are the consistent:
    # E1A: 1.87951932
    # E1R: 2.79688494

    assert np.isclose(y[0], 1.87951932, atol=1e-4)
    assert np.isclose(y[1], 2.79688494, atol=1e-4)
