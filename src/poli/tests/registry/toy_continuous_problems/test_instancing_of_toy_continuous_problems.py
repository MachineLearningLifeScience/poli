"""This module tests the instancing of toy continuous problems."""


def test_create_ackley_function_01():
    """Tests the instancing of the Ackley function in 2D."""
    from poli import objective_factory

    _, f_ackley, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=2,
    )


def test_create_ackley_function_01_on_more_dimensions():
    """Tests the instancing of the Ackley function in 2D."""
    from poli import objective_factory

    _, f_ackley, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
