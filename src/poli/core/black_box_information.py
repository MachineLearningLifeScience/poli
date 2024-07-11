"""
Implements the black box information class.

Black boxes come in many shapes and forms. The information contained in this class
is used to describe the black box and its properties.

The black box information includes the following information:
- The problem's name.
- The length of the longest sequence.
- Whether the sequences need to be aligned.
- Whether the sequences need to have a fixed length.
- The alphabet of allowed characters.
"""

from typing import Literal, Union

import numpy as np


class BlackBoxInformation:
    def __init__(
        self,
        name: str,
        max_sequence_length: int,
        aligned: bool,
        fixed_length: bool,
        deterministic: bool,
        alphabet: list,
        log_transform_recommended: bool = None,
        discrete: bool = True,
        fidelity: Union[Literal["high", "low"], None] = None,
        padding_token: str = "",
    ):
        self.name = name
        self.max_sequence_length = max_sequence_length
        self.aligned = aligned
        self.fixed_length = fixed_length
        self.deterministic = deterministic
        self.alphabet = alphabet
        self.log_transform_recommended = log_transform_recommended
        self.discrete = discrete
        self.fidelity = fidelity
        self.padding_token = padding_token

    def get_problem_name(self) -> str:
        """Returns the problem's name.

        Returns
        --------
        name : str
            The problem's name.
        """
        return self.name

    def get_max_sequence_length(self) -> int:
        """
        Returns the maximum sequence length allowed by the black-box.

        Returns
        --------
        max_sequence_length : int
            The length of the longest sequence.
        """
        return self.max_sequence_length

    def is_deterministic(self) -> bool:
        """
        Returns whether the black-box is deterministic.

        Returns
        --------
        deterministic : bool
            Whether the black-box is deterministic.
        """
        return self.deterministic

    def is_discrete(self) -> bool:
        """
        Returns whether the black-box has discrete inputs.

        Returns
        --------
        discrete : bool
            Whether the black-box has discrete inputs.
        """
        return self.discrete

    def get_padding_token(self) -> str:
        """
        Returns the padding token used by the black-box.

        By default, this is usually "", the empty string.

        Returns
        --------
        padding_token : str
            The padding token used by the black-box.
        """
        return self.padding_token

    def sequences_are_aligned(self) -> bool:
        """
        Returns whether the sequences need to be aligned.

        We defined aligned sequences as sequences that are aligned in such
        a way that the same position in each sequence corresponds to the
        same position in the other sequences.

        Problems can be aligned, but have several different lengths. One example
        is the protein sequence problem for several different wildtypes.

        Returns
        --------
        aligned : bool
            Whether the sequences need to be aligned.
        """
        return self.aligned

    def get_alphabet(self) -> list:
        """
        Returns the alphabet of allowed characters.

        Returns
        --------
        alphabet : list[str]
            List of tokens allowed by the black-box.
        """
        return self.alphabet

    def log_transform_recommended(self) -> bool:
        """
        Returns whether the black-box recommends log-transforming the targets.

        Returns
        --------
        log_transform_recommended : bool
            Whether the black-box recommends log-transforming the targets.
        """
        return self.log_transform_recommended

    def __str__(self):
        return f"BlackBoxInformation(name={self.name}, max_sequence_length={self.max_sequence_length}, aligned={self.aligned}, fixed_length={self.fixed_length}, discrete={self.discrete})"

    def __repr__(self):
        return f"<BlackBoxInformation(name={self.name}, max_sequence_length={self.max_sequence_length}, aligned={self.aligned}, fixed_length={self.fixed_length}, discrete={self.discrete}, alphabet={self.alphabet}, log_transform_recommended={self.log_transform_recommended})>"

    def as_dict(self):
        return {
            "name": self.name,
            "max_sequence_length": (
                self.max_sequence_length
                if self.max_sequence_length != np.inf
                else "inf"
            ),
            "aligned": self.aligned,
            "fixed_length": self.fixed_length,
            "deterministic": self.deterministic,
            "discrete": self.discrete,
            "fidelity": self.fidelity,
            "alphabet": self.alphabet,
            "log_transform_recommended": self.log_transform_recommended,
            "padding_token": self.padding_token,
        }
