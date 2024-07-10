"""
This is a suite of tests for the Ehrlich functions
proposed by Stanton et al. (2024).
"""

import numpy as np

import pytest

from poli.objective_repository.ehrlich._construct_feasibility_matrix import (
    _construct_sparse_transition_matrix,
)


@pytest.mark.parametrize("size", [3, 5, 8, 10])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_sparse_matrix_construction_is_ergodic_and_aperiodic(size: int, seed: int):
    sparse_transition_matrix = _construct_sparse_transition_matrix(size, seed=seed)

    assert (
        np.linalg.matrix_power(sparse_transition_matrix, (size - 1) ** 2 + 1) > 0.0
    ).all()
