"""Tests embedding continuous toy objectives into higher dimensions.

For some objective functions that are defined in 2D
space, we give the users the affordance to embed them
in higher dimensions. This is useful for testing
higher dimensional Bayesian Optimization algorithms,
since some of them assume that the intrinsic dimensionality
of the problem is lower than the actual dimensionality.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from poli.objective_repository.toy_continuous_problem.register import (
    ToyContinuousBlackBox,
)


def test_embed_camelback_into_high_dimensions():
    from poli import objective_factory

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="camelback_2d",
        n_dimensions=2,
        embed_in=10,
    )
    f_camelback = cast(ToyContinuousBlackBox, problem.black_box)

    dimensions_to_embed_in = f_camelback.function.dimensions_to_embed_in

    # Testing whether the output is the same as long as we
    # are in the same subspace.
    one_x = np.random.randn(10).reshape(1, -1)
    another_x = np.random.randn(10).reshape(1, -1)

    one_x[0, dimensions_to_embed_in] = [0.0, 0.0]
    another_x[0, dimensions_to_embed_in] = [0.0, 0.0]

    assert np.allclose(
        f_camelback(one_x),  # type: ignore
        f_camelback(another_x),  # type: ignore
    )

    # Testing whether the output is different if we are
    # in different subspaces.
    one_x[0, dimensions_to_embed_in] = [1.0, 1.0]

    assert not np.allclose(
        f_camelback(one_x),  # type: ignore
        f_camelback(another_x),  # type: ignore
    )


if __name__ == "__main__":
    test_embed_camelback_into_high_dimensions()
