"""This module tests the instancing of toy continuous problems."""

import pytest

import numpy as np

from poli.objective_repository.toy_continuous_problem.register import POSSIBLE_FUNCTIONS


@pytest.mark.parametrize("function_name", POSSIBLE_FUNCTIONS)
def test_create_ackley_function_01(function_name):
    """Tests the instancing of the Ackley function in 2D."""
    from poli import objective_factory

    _, f, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name=function_name,
        n_dimensions=2,
    )

    f(np.array([[0.0, 0.0]]))


def test_create_ackley_function_01_on_more_dimensions():
    """Tests the instancing of the Ackley function in 2D."""
    from poli import objective_factory

    _, f_ackley, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )