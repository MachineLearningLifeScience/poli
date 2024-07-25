"""
Implements the problem setup information, which contains the problem information (e.g. alphabet, sequence length...).
"""

from poli.core.black_box_information import BlackBoxInformation


class ProblemSetupInformation:
    def __init__(
        self,
        name: str,
        black_box_information: BlackBoxInformation,
    ):
        """
        Initialize the ProblemSetupInformation object.

        Parameters
        ----------
        name : str
            The problem's name.
        """
        self.name = name
        self.black_box_information = black_box_information

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

    def sequences_are_aligned(self) -> bool:
        """
        Returns whether the sequences need to be aligned.

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

    def as_dict(self) -> dict:
        """Returns all attributes as a dictionary.

        Returns
        --------
        info : dict
            A dictionary of all attributes.

        Notes
        -----
        - It's vital that this method is _not_ called __dict__, since
          that breaks the serialization of the ProblemSetupInformation
          when using pickle. (see https://stackoverflow.com/a/75777082/3516175)
        """
        return {
            "name": self.name,
            "max_sequence_length": self.max_sequence_length,
            "aligned": self.aligned,
            "alphabet": self.alphabet,
            "log_transform_recommended": self.log_transform_recommended,
        }

    def __str__(self) -> str:
        return f"ProblemSetupInformation(name={self.name})"

    def __repr__(self) -> str:
        return f"<ProblemSetupInformation(name={self.name}, max_sequence_length={self.max_sequence_length}, aligned={self.aligned}, alphabet={self.alphabet}, log_transform_recommended={self.log_transform_recommended})>"
