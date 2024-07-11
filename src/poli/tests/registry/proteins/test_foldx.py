from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

HOME_DIR = Path().home().resolve()
PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
if not PATH_TO_FOLDX_FILES.exists():
    pytest.skip("FoldX is not installed. ", allow_module_level=True)

if not (PATH_TO_FOLDX_FILES / "foldx").exists():
    pytest.skip("FoldX is not compiled. ", allow_module_level=True)


@pytest.mark.poli__protein
def test_running_foldx_stability():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    problem = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 32.4896).all()


@pytest.mark.poli__protein
def test_running_foldx_sasa():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    problem = objective_factory.create(
        name="foldx_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 8411.45578009).all()


@pytest.mark.poli__protein
def test_running_foldx_stability_and_sasa():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    problem = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0[:, 0], 32.4896).all()
    assert np.isclose(y0[:, 1], 8411.45578009).all()


@pytest.mark.poli__protein
def test_registering_foldx_stability_and_sasa_with_verbose_output():
    """
    Testing whether the foldx output is printed.
    """
    problem = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=THIS_DIR / "101m_Repair.pdb",
        verbose=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0[:, 0], 32.4896).all()
    assert np.isclose(y0[:, 1], 8411.45578009).all()


@pytest.mark.poli__protein
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
    problem = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=wildtype_pdb_path,
        eager_repair=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 32.6135).all()


@pytest.mark.poli__protein
def test_foldx_from_repaired_file():
    """
    In this test, we check whether no repair is
    performed if the file already contains _Repair.
    """
    wildtype_pdb_path = THIS_DIR / "101m_Repair.pdb"
    problem = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=wildtype_pdb_path,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 32.4896).all()
