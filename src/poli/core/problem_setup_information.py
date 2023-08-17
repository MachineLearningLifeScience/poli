from pathlib import Path


class ProblemSetupInformation:
    def __init__(
        self,
        name: str,
        max_sequence_length: int,
        aligned: bool,
        alphabet: dict,
        log_transform_recommended=False,
    ):
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

    def get_alphabet(self) -> dict:
        return self.alphabet

    def log_transform_recommended(self) -> bool:
        return self.log_transform_recommended
