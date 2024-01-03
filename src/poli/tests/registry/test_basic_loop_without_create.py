"""This test module exemplifies the basic loop of creating
objective functions without relying on create.

Not using objective_factory.create comes with the added
benefit of having IDE IntelliSense support.
"""


def test_basic_loop():
    from poli.objective_repository import ToyContinuousProblemFactory

    problem_factory = ToyContinuousProblemFactory()

    f, x0, y0 = problem_factory.create(function_name="ackley_function_01")
