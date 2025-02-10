from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

HOME_DIR = Path().home().resolve()


@pytest.mark.poli__dms
def test_running_gb1_fitness():
    """
    Testing whether we can register the GB1 problem.
    """
    problem = objective_factory.create(
        name="dms_gb1",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 1.0).all()


@pytest.mark.poli__dms
def test_running_gb1_negate_fitness():
    """
    Testing whether we can register the GB1 problem.
    """
    problem = objective_factory.create(
        name="dms_gb1",
        negative=True,
    )
    f, x = problem.black_box, problem.x0
    x[0, 0] = "Q"
    x[0, 1] = "R"
    x[0, 2] = "L"
    x[0, 3] = "G"
    y = f(x)

    assert np.isclose(y, -2.727247889).all()


@pytest.mark.poli__dms
def test_running_trpb_fitness():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    problem = objective_factory.create(
        name="dms_trpb",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, 0.408073925).all()


@pytest.mark.poli__dms
def test_running_trpb_negative_fitness():
    """
    Testing whether we can register the logp problem
    if biopython and python-levenshtein are installed.
    """
    problem = objective_factory.create(
        name="dms_trpb",
        negative=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    assert np.isclose(y0, -0.408073925).all()
