"""Utilities for transforming levels to arrays and back."""

from itertools import product
from typing import List

import numpy as np


def level_to_list(level_txt: str) -> List[List[str]]:
    """
    Takes a level as a string and returns
    a list of lists of individual tokens.
    """
    # Returns a list by splitting the level text
    # by \n.
    as_list = level_txt.split("\n")
    return [list(row) for row in as_list if row != ""]


def level_to_array(level_txt: str) -> np.ndarray:
    """Parses a level from string to numpy array."""
    # Returns a np array by splitting the level text
    # by \n.
    return np.array(level_to_list(level_txt))


def levels_to_onehot(levels: np.ndarray, n_sprites: int = 11) -> np.ndarray:
    """Transforms an array [b, w, h] of integers into a one-hot array [b, n_sprites, w, h]."""
    batch_size, w, h = levels.shape
    y_onehot = np.zeros((batch_size, n_sprites, h, w))
    for b, level in enumerate(levels):
        # for loop through the batch, no?
        # print(level)
        for i, j in product(range(h), range(w)):
            c = level[i, j]
            y_onehot[b, c, i, j] = 1

    return y_onehot


def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]


def onehot_to_levels(levels_onehot: np.ndarray, sampling=False, seed=0) -> np.ndarray:
    """
    Transforms a level from probits to integers.
    """
    if sampling:
        # From log-softmax to softmax.
        levels_onehot = np.exp(levels_onehot)
        np.random.seed(seed)
        batch_size, n_classes, h, w = levels_onehot.shape
        # u = np.random.rand(n_classes)
        # U = np.zeros_like(levels_onehot)
        # for b in range(batch_size):
        #     for i, j in product(range(h), range(w)):
        #         U[b, :, i, j] = u
        # levels = (levels_onehot.cumsum(axis=1) > U).argmax(axis=1)

        # Is there a smarter way to do this?
        # There is:
        # https://stackoverflow.com/a/34190035/3516175
        levels = np.zeros((batch_size, h, w), dtype=int)
        for b in range(batch_size):
            for i, j in product(range(h), range(w)):
                p = levels_onehot[b, :, i, j]
                levels[b, i, j] = np.random.choice(n_classes, p=p)

        np.random.seed()
    else:
        levels = np.argmax(levels_onehot, axis=1)

    return levels


def add_padding_to_level(level: np.ndarray, n_padding: int = 1) -> np.ndarray:
    """
    Adds padding to the left of the level, giving room
    for the agent to land.
    """
    h, w = level.shape
    padding = 2 * np.ones((h, n_padding))  # Starting with emptyness.
    padding[-1, :] = 0  # Adding the ground.
    level_with_padding = np.concatenate((padding, level), axis=1)

    return level_with_padding


def clean_level(level: np.ndarray) -> List[List[int]]:
    """
    Cleans a level by removing Mario (token id: 11),
    and replacing it with empty space.
    """
    # Cleaning up Mario (11), replacing
    # it with empty space (2).
    level[level == 11] = 2
    level = level.astype(int)
    # level = add_padding_to_level(level, 2)

    return level.tolist()
