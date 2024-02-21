import numpy as np


def test_instancing_black_boxes_alone():
    from poli.objective_repository import WhiteNoiseBlackBox

    f = WhiteNoiseBlackBox()

    f(np.array([["1", "2", "3"]]))
