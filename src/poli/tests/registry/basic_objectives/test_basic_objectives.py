from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


def test_registering_white_noise():
    _, f, _, _, _ = objective_factory.create(name="white_noise")
    x = np.array([["A", "B", "C", "D"]])
    f(x)
    f.terminate()


def test_registering_aloha():
    _, f, _, y0, _ = objective_factory.create(name="aloha")
    x = np.array([list("ALOOF")])
    assert f(x) == 3
    f.terminate()
