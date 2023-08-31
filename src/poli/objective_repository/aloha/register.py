"""
This is a registration script for the ALOHA problem,
a simple example of a discrete black box objective
function where the goal is to find the sequence
["A", "L", "O", "H", "A"] among all 5-letter sequences.

The problem is registered as 'aloha', and it uses
a conda environment called 'poli__base' (see the
environment.yml file in this folder).
"""
from typing import Tuple, Dict
from string import ascii_uppercase

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation


class AlohaBlackBox(AbstractBlackBox):
    def __init__(self, info: ProblemSetupInformation, batch_size: int = None):
        self.alphabet = {symbol: idx for idx, symbol in enumerate(info.alphabet)}
        super().__init__(info, batch_size)

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        # x is a [b, L] array of strings or ints, if they are
        # ints, then we should convert them to strings
        # using the alphabet.
        # TODO: this assumes that the input is a batch of size 1.
        # Address this when we change __call__.
        if x.dtype.kind == "i":
            if self.alphabet is None:
                raise ValueError(
                    "The alphabet must be defined if the input is an array of ints."
                )
            inverse_alphabet = {v: k for k, v in self.alphabet.items()}
            x = np.array([[inverse_alphabet[i] for i in x[0]]])

        matches = x == np.array(["A", "L", "O", "H", "A"])
        return np.sum(matches.reshape(-1, 1), axis=0, keepdims=True)


class AlohaProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        # The alphabet: ["A", "B", "C", ...]
        alphabet = list(ascii_uppercase)

        return ProblemSetupInformation(
            name="aloha",
            max_sequence_length=5,
            aligned=True,
            alphabet=alphabet,
        )

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        problem_info = self.get_setup_information()
        f = AlohaBlackBox(info=problem_info)
        x0 = np.array([["A", "L", "O", "O", "F"]])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    aloha_problem_factory = AlohaProblemFactory()
    register_problem(
        aloha_problem_factory,
        conda_environment_name="poli__base",
        # force=True
    )
