"""This module tests whether forcing the isolation of black
box objectives indeed spawns an isolated process."""


def test_force_isolation():
    from poli import objective_factory

    problem = objective_factory.create(
        name="deco_hop",
        force_isolation=True,
    )

    assert isinstance(problem.black_box, objective_factory.ExternalBlackBox)
