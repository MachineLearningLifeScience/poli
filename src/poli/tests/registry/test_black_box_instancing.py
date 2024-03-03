import pytest

import numpy as np

from poli.objective_repository import (
    AlohaBlackBox,
    DockstringBlackBox,
    DRD3BlackBox,
    WhiteNoiseBlackBox,
)

test_data = [
    ("aloha", AlohaBlackBox, {}),
    (
        "dockstring",
        DockstringBlackBox,
        {"target_name": "drd2", "string_representation": "SMILES"},
    ),
    (
        "drd3_docking",
        DRD3BlackBox,
        {"string_representation": "SMILES", "force_isolation": True},
    ),
    # ("white_noise", WhiteNoiseBlackBox, {}),
]


@pytest.mark.parametrize(
    "black_box_name, black_box_class, kwargs_for_black_box",
    test_data,
)
def test_instancing_a_black_box_both_ways_matches(
    black_box_name, black_box_class, kwargs_for_black_box
):
    from poli import create_problem

    problem = create_problem(
        name=black_box_name,
        seed=42,
        evaluation_budget=100,
        **kwargs_for_black_box,
    )
    x0 = problem.x0
    y0 = problem.black_box(x0)

    f = black_box_class(**kwargs_for_black_box)

    assert np.allclose(f(x0), y0)


if __name__ == "__main__":
    for black_box_name, black_box_class, kwargs in test_data:
        test_instancing_a_black_box_both_ways_matches(
            black_box_name, black_box_class, kwargs
        )
