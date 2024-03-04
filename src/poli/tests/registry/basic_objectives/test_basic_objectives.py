from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


def test_registering_white_noise():
    white_noise_problem = objective_factory.create(name="white_noise")
    f, x0 = white_noise_problem.black_box, white_noise_problem.x0
    y0 = f(x0)
    f.terminate()


def test_registering_aloha():
    f, _, y0 = objective_factory.create(name="aloha")
    x = np.array([list("ALOOF")])
    assert f(x) == 3
    f.terminate()
