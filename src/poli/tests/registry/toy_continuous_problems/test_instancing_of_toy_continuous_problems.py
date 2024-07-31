"""This module tests the instancing of toy continuous problems."""

import numpy as np
import pytest

from poli.objective_repository.toy_continuous_problem.toy_continuous_problem import (
    POSSIBLE_FUNCTIONS,
    SIX_DIMENSIONAL_PROBLEMS,
    TWO_DIMENSIONAL_PROBLEMS,
)


@pytest.mark.parametrize("function_name", POSSIBLE_FUNCTIONS)
def test_create_toy_objective_function(function_name):
    """Tests the instancing the given objective function."""
    from poli import objective_factory

    if function_name in SIX_DIMENSIONAL_PROBLEMS:
        n_dimensions = 6
    elif function_name in TWO_DIMENSIONAL_PROBLEMS:
        n_dimensions = 2
    else:
        n_dimensions = 10

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name=function_name,
        n_dimensions=n_dimensions,
    )
    f = problem.black_box
    x0 = problem.x0

    f(x0)


def test_create_ackley_function_01_on_more_dimensions():
    """Tests the instancing of the Ackley function in 2D."""
    from poli import objective_factory

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    f_ackley = problem.black_box
    f_ackley(np.array([[0.0] * 10]))
