"""This module tests whether forcing the isolation of black
box objectives indeed spawns an isolated process."""


def test_force_isolation_on_aloha():
    from poli import objective_factory

    f, _, _ = objective_factory.create_problem(
        name="aloha",
        force_register=True,
        force_isolation=True,
    )

    assert isinstance(f, objective_factory.ExternalBlackBox)


if __name__ == "__main__":
    test_force_isolation_on_aloha()
