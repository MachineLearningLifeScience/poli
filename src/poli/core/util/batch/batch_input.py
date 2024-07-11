"""This module implements a batched input generator."""

from itertools import islice
from typing import Iterable


def batched(iterable: Iterable, chunk_size: int):
    """
    Generate batches of elements from an iterable.

    If working on Python 3.12+, this function can be replaced
    with the built-in function `batched` from `itertools`.
    https://docs.python.org/3/library/itertools.html#itertools.batched

    Parameters
    ----------
    iterable : Iterable
        The iterable to be batched.
    chunk_size : int
        The size of each batch.

    Yields
    ------
    tuple
        A tuple containing elements from the iterable, with each tuple having a length equal to chunk_size.

    Examples
    --------
    >>> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> for batch in batched(numbers, 3):
    ...     print(batch)
    (1, 2, 3)
    (4, 5, 6)
    (7, 8, 9)
    (10,)

    """
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk
