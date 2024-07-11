from __future__ import annotations

import numpy as np


def _construct_banded_matrix(size: int) -> np.ndarray:
    """
    Constructs a matrix of zeroes and ones, where
    the ones are bands that can loop around.
    """
    matrix = np.zeros((size, size), dtype=int)
    band_index = 0
    band_length = size - 1
    for row_i in range(size):
        indices_for_positions_that_will_be_1 = list(
            range(band_index, band_index + band_length)
        )

        # Looping the ones that go over the limit
        for i in range(len(indices_for_positions_that_will_be_1)):
            if indices_for_positions_that_will_be_1[i] >= size:
                indices_for_positions_that_will_be_1[i] -= size

        matrix[row_i, indices_for_positions_that_will_be_1] = 1

        band_index += 1

    return matrix


def _construct_binary_mask(size: int) -> np.ndarray:
    banded_matrix = _construct_banded_matrix(size)

    # Shuffle its rows
    random_indices_for_rows = np.random.permutation(size)
    binary_mask_matrix = banded_matrix[random_indices_for_rows]

    # Making sure that the diagonal is full
    # of ones
    binary_mask_matrix[np.diag_indices(size)] = 1

    return binary_mask_matrix


def _construct_transition_matrix(size: int, seed: int | None = None) -> np.ndarray:
    binary_mask_matrix = _construct_binary_mask(size)

    # Creating a random state and matrix
    random_state = np.random.RandomState(seed)
    random_matrix = random_state.randn(size, size)

    # Softmax it
    transition_matrix = np.exp(random_matrix) / np.sum(np.exp(random_matrix), axis=0)

    # Mask it
    masked_transition_matrix = transition_matrix * binary_mask_matrix

    # Normalize it
    normalized_transition_matrix = masked_transition_matrix / np.sum(
        masked_transition_matrix, axis=1, keepdims=True
    )

    return normalized_transition_matrix
