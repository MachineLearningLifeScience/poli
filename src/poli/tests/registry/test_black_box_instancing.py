import pytest

import numpy as np

from poli.objective_repository import (
    AlohaBlackBox,
    DockstringBlackBox,
    DRD3BlackBox,
    FoldxRFPLamboBlackBox,
    WhiteNoiseBlackBox,
)

SEED = np.random.randint(0, 1000)

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
    (
        "foldx_rfp_lambo",
        FoldxRFPLamboBlackBox,
        {},
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
    from poli import create
    from poli.core.util.seeding import seed_python_numpy_and_torch

    problem = create(
        name=black_box_name,
        seed=SEED,
        **kwargs_for_black_box,
    )
    x0 = problem.x0
    if black_box_name == "foldx_rfp_lambo":
        x0 = x0[0].reshape(1, -1)
    y0 = problem.black_box(x0)

    seed_python_numpy_and_torch(SEED)
    f = black_box_class(**kwargs_for_black_box)
    y0_ = f(x0)

    assert np.allclose(y0_, y0)


if __name__ == "__main__":
    for black_box_name, black_box_class, kwargs in test_data:
        test_instancing_a_black_box_both_ways_matches(
            black_box_name, black_box_class, kwargs
        )
