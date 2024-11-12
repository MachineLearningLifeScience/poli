from __future__ import annotations

import numpy as np


def _construct_banded_matrix(size: int, band_length: int | None = None) -> np.ndarray:
    """
    Constructs a matrix of zeroes and ones, where
    the ones are bands that can loop around.

    The length of the non-zero band is given by band_length.
    By default, it is set to size - 2 * (size // 5). This
    value is taken from the original Ehrlich paper [1].

    References
    ----------
    [1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
    Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
    arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
    """
    matrix = np.zeros((size, size), dtype=int)
    band_index = 0
    band_length = size - ((2 * size) // 5) if band_length is None else band_length
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


def _construct_binary_mask(size: int, band_length: int | None = None) -> np.ndarray:
    banded_matrix = _construct_banded_matrix(size, band_length=band_length)

    # Shuffle its rows
    random_indices_for_rows = np.random.permutation(size)
    binary_mask_matrix = banded_matrix[random_indices_for_rows]

    # Making sure that the diagonal is full
    # of ones
    binary_mask_matrix[np.diag_indices(size)] = 1

    return binary_mask_matrix


def _construct_transition_matrix(
    size: int,
    seed: int | None = None,
    temperature: float = 0.5,
    band_length: int | None = None,
) -> np.ndarray:
    binary_mask_matrix = _construct_binary_mask(size, band_length=band_length)

    # Creating a random state and matrix
    random_state = np.random.RandomState(seed)
    random_matrix = random_state.randn(size, size)

    # Softmax it with low temperature
    transition_matrix = np.exp(random_matrix / temperature) / np.sum(
        np.exp(random_matrix / temperature), axis=0
    )

    # Mask it
    masked_transition_matrix = transition_matrix * binary_mask_matrix

    # Normalize it
    normalized_transition_matrix = masked_transition_matrix / np.sum(
        masked_transition_matrix, axis=1, keepdims=True
    )

    return normalized_transition_matrix
