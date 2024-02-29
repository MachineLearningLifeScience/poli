"""Tests for the budget exhaustion inside objective functions."""

import pytest

import numpy as np

from poli.core.exceptions import BudgetExhaustedException


def test_budget_exhaustion_exception():
    """Test that the exception is raised when the budget is exhausted."""
    with pytest.raises(BudgetExhaustedException):
        raise BudgetExhaustedException()


def test_num_evaluation_tracks_correctly():
    from poli import objective_factory

    f, _, _ = objective_factory.create_problem(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=3,
        evaluation_budget=10,
    )

    assert f.num_evaluations == 1

    f.reset_evaluation_budget()

    x0 = np.array([[0.0, 0.0, 0.0]] * 9)

    f(x0)

    assert f.num_evaluations == 9


def test_budget_exhausts():
    from poli import objective_factory

    f, _, _ = objective_factory.create_problem(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=3,
        evaluation_budget=10,
    )

    f.reset_evaluation_budget()

    x0 = np.array([[0.0, 0.0, 0.0]] * 9)

    f(x0)

    with pytest.raises(BudgetExhaustedException):
        f(x0)


if __name__ == "__main__":
    test_budget_exhausts()
