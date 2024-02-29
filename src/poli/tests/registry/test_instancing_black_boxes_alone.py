import numpy as np

from poli.objective_repository import AVAILABLE_BLACK_BOXES


def test_instancing_black_boxes_alone():
    from poli.objective_repository import WhiteNoiseBlackBox

    f = WhiteNoiseBlackBox()

    f(np.array([["1", "2", "3"]]))


def test_instancing_black_boxes_alone_in_isolation():
    from poli.objective_repository import IsolatedBlackBox

    f = IsolatedBlackBox(
        name="dockstring",
        target_name="abl1",
    )
