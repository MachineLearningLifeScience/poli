"""
This test module tests whether the seeding works
in the white noise example.
"""

import numpy as np

from poli import objective_factory


def test_seeding_in_white_noise_factory_creation():
    problem_1 = objective_factory.create(
        "white_noise", seed=42, batch_size=1, parallelize=False
    )
    f, x0 = problem_1.black_box, problem_1.x0
    y0 = f(x0)

    problem_2 = objective_factory.create(
        "white_noise", seed=42, batch_size=1, parallelize=False
    )
    f, x0 = problem_2.black_box, problem_2.x0
    y1 = f(x0)

    problem_3 = objective_factory.create(
        "white_noise", seed=43, batch_size=1, parallelize=False
    )
    f, x0 = problem_3.black_box, problem_3.x0
    y2 = f(x0)

    assert (y0 == y1).all() and not np.isclose(y0, y2).all()
