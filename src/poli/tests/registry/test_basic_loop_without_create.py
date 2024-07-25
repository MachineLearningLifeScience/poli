"""This test module exemplifies the basic loop of creating
objective functions without relying on create.

Not using objective_factory.create comes with the added
benefit of having IDE IntelliSense support.
"""


def test_basic_loop_without_create():
    from poli.objective_repository import ToyContinuousProblemFactory

    problem_factory = ToyContinuousProblemFactory()

    problem = problem_factory.create(function_name="ackley_function_01")  # noqa F841


def test_instancing_black_boxes_alone():
    from poli.objective_repository import ToyContinuousBlackBox

    f = ToyContinuousBlackBox(function_name="ackley_function_01")  # noqa F841
