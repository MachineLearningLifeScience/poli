"""Tests embedding continuous toy objectives into higher dimensions.

For some objective functions that are defined in 2D
space, we give the users the affordance to embed them
in higher dimensions. This is useful for testing
higher dimensional Bayesian Optimization algorithms,
since some of them assume that the intrinsic dimensionality
of the problem is lower than the actual dimensionality.
"""

import numpy as np


def test_embed_camelback_into_high_dimensions():
    from poli import objective_factory
    from poli.objective_repository.toy_continuous_problem.register import (
        ToyContinuousProblem,
    )

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="camelback_2d",
        n_dimensions=2,
        embed_in=10,
    )
    f_camelback: ToyContinuousProblem = problem.black_box

    dimensions_to_embed_in = f_camelback.function.dimensions_to_embed_in

    # Testing whether the output is the same as long as we
    # are in the same subspace.
    one_x = np.random.randn(10).reshape(1, -1)
    another_x = np.random.randn(10).reshape(1, -1)

    one_x[0, dimensions_to_embed_in] = [0.0, 0.0]
    another_x[0, dimensions_to_embed_in] = [0.0, 0.0]

    assert np.allclose(
        f_camelback(one_x),
        f_camelback(another_x),
    )

    # Testing whether the output is different if we are
    # in different subspaces.
    one_x[0, dimensions_to_embed_in] = [1.0, 1.0]

    assert not np.allclose(
        f_camelback(one_x),
        f_camelback(another_x),
    )
