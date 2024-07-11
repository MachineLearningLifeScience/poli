import numpy as np


def test_multi_objective_instantiation():
    from poli.core.multi_objective_black_box import MultiObjectiveBlackBox
    from poli.objective_repository import AlohaBlackBox

    f_aloha = AlohaBlackBox()

    f = MultiObjectiveBlackBox(
        objective_functions=[f_aloha, f_aloha],
    )

    assert f.objective_functions == [f_aloha, f_aloha]

    x0 = np.array([["A", "B", "C", "D", "E"]])
    y0 = f(x0)

    assert y0.shape == (1, 2)


def test_negative_black_boxes():
    from poli.objective_repository import AlohaBlackBox

    f = AlohaBlackBox()
    g = -f

    x0 = np.array([["A", "B", "C", "D", "E"]])

    f0 = f(x0)
    g0 = g(x0)

    assert f0 == -g0
