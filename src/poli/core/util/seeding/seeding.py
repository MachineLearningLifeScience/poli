import numpy as np
import random


def seed_numpy(seed: int = None) -> None:
    """
    Seed the NumPy random number generator.

    Parameters:
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, the generator is initialized with a random seed.

    Examples:
    ---------
    >>> seed_numpy(123)
    """
    np.random.seed(seed)


def seed_python(seed: int = None) -> None:
    """
    Seed the random number generator for Python.


    Parameters:
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, the generator is initialized with a random seed.
    """
    random.seed(seed)
