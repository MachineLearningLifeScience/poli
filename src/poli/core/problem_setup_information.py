from typing import List, Dict, Union

from pathlib import Path


class ProblemSetupInformation:
    def __init__(
        self,
        name: str,
        max_sequence_length: int,
        aligned: bool,
        alphabet: List[str],
        log_transform_recommended=False,
    ):
        """
        Initialize the ProblemSetupInformation object.

        Parameters
        ----------
        name : str
            The problem's name.
        max_sequence_length : int
            The length of the longest sequence.
        aligned : bool
            Whether the sequences have been aligned.
        alphabet : List[str]
            List of characters that may appear.
        log_transform_recommended : bool, optional
            A recommendation for optimization algorithm whether to log transform the targets.
            Default is False.
        """
        self.name = name
        self.max_sequence_length = max_sequence_length
        self.aligned = aligned
        self.alphabet = alphabet
        self.log_transform_recommended = log_transform_recommended

    def get_problem_name(self) -> str:
        return self.name

    def get_max_sequence_length(self) -> int:
        return self.max_sequence_length

    def sequences_are_aligned(self) -> bool:
        return self.aligned

    def get_alphabet(self) -> list:
        return self.alphabet

    def log_transform_recommended(self) -> bool:
        return self.log_transform_recommended

    def as_dict(self) -> dict:
        """Returns all attributes as a dictionary.

        Returns:
        --------
        dict
            A dictionary of all attributes.
        """
        return {
            "name": self.name,
            "max_sequence_length": self.max_sequence_length,
            "aligned": self.aligned,
            "alphabet": self.alphabet,
            "log_transform_recommended": self.log_transform_recommended,
        }
