"""This test suite contains the core promises we make to the user."""

import numpy as np


def test_creating_an_instance_of_a_black_box():
    from poli.objective_repository import WhiteNoiseBlackBox

    f = WhiteNoiseBlackBox()
    x = np.array([["1", "2", "3"]])
    _ = f(x)


def test_creating_a_problem():
    from poli import create

    white_noise_problem = create(
        name="white_noise",
        seed=42,
        evaluation_budget=100,
    )

    f, x0 = white_noise_problem.black_box, white_noise_problem.x0
    _ = f(x0)

    f.terminate()
