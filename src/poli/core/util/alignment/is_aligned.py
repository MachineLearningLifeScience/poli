"""Utility model to check if the input to an "aligned"
problem is indeed aligned.
"""

import numpy as np


def is_aligned_input(x: np.ndarray, maximum_sequence_length: int = None) -> bool:
    """Utility function to check if the input to an "aligned"
    problem is indeed aligned.

    Our definition of alignment goes as follows:
    - The input is of shape [b, L], where L is a defined maximum
      length.
    - If the input is of shape [b,], then the lengths of all
      elements inside the batch are the same.
    """
    if maximum_sequence_length == np.inf:
        maximum_sequence_length = None

    # If the array is 2-dimensional, we only need to check
    # if the second dimension is equal to the maximum sequence length
    # (when provided)
    if x.ndim == 2:
        if maximum_sequence_length is not None:
            return x.shape[1] == maximum_sequence_length
        else:
            return True
    # If the array is 1-dimensional, we need to check if all
    # elements have the same length
    elif x.ndim == 1:
        unique_lengths = set(map(len, x))
        if len(unique_lengths) == 1:
            if maximum_sequence_length is not None:
                return unique_lengths.pop() == maximum_sequence_length
            else:
                return True
        else:
            # Then there's more than one length
            return False
    else:
        return False
