"""Utilities for seeding random number generators."""

import random

import numpy as np


def seed_numpy(seed: int = None) -> None:
    """
    Seed the NumPy random number generator.

    Parameters
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, then no seeding is performed.
    """
    if seed is not None:
        np.random.seed(seed)


def seed_python(seed: int = None) -> None:
    """
    Seed the random number generator for Python.


    Parameters
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, then no seeding is performed.
    """
    if seed is not None:
        random.seed(seed)


def seed_torch(seed: int = None) -> None:
    """
    Seed the random number generator for PyTorch.


    Parameters
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, no seeding is performed.
    """
    try:
        import torch
    except ImportError:
        return

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def seed_python_numpy_and_torch(seed: int = None) -> None:
    """
    Seed all random number generators.


    Parameters
    ----------
    seed : int, optional
        Seed value for the random number generator. If None, no seeding is performed.
    """
    seed_numpy(seed)
    seed_python(seed)
    seed_torch(seed)
