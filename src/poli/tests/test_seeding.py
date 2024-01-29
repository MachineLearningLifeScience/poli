"""
This test module tests whether the seeding works
in the white noise example.
"""

import numpy as np

from poli import objective_factory


def test_seeding_in_white_noise():
    _, _, _, y0, _ = objective_factory.create(
        "white_noise", seed=42, batch_size=1, parallelize=False
    )

    _, _, _, y1, _ = objective_factory.create(
        "white_noise", seed=42, batch_size=1, parallelize=False
    )

    _, _, _, y2, _ = objective_factory.create(
        "white_noise", seed=43, batch_size=1, parallelize=False
    )

    assert (y0 == y1).all() and not np.isclose(y0, y2).all()


if __name__ == "__main__":
    test_seeding_in_white_noise()
