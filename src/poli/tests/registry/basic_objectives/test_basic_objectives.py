import numpy as np

from poli import objective_factory


def test_registering_white_noise():
    white_noise_problem = objective_factory.create(name="white_noise")
    f, x0 = white_noise_problem.black_box, white_noise_problem.x0
    _ = f(x0)
    f.terminate()


def test_registering_aloha():
    aloha_problem = objective_factory.create(name="aloha")
    f = aloha_problem.black_box
    x = np.array([list("ALOOF")])
    assert f(x) == 3
    f.terminate()
