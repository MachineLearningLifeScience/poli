"""Tests for the budget exhaustion inside objective functions."""

import numpy as np
import pytest

from poli.core.exceptions import BudgetExhaustedException


def test_num_evaluation_tracks_correctly():
    from poli import objective_factory

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=3,
        evaluation_budget=10,
    )
    f = problem.black_box

    assert f.num_evaluations == 0

    f.reset_evaluation_budget()

    x0 = np.array([[0.0, 0.0, 0.0]] * 9)

    f(x0)

    assert f.num_evaluations == 9


def test_budget_exhausts():
    from poli import objective_factory

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=3,
        evaluation_budget=10,
    )
    f = problem.black_box

    x0 = np.array([[0.0, 0.0, 0.0]] * 9)

    f(x0)

    with pytest.raises(BudgetExhaustedException):
        f(x0)
