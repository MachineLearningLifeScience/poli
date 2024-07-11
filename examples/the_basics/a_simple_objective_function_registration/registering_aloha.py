"""
This is the minimal example of how to register
a problem factory, which allows for creating
instances of the problem: the objective function,
the initial point, and its first evaluation.
"""

from string import ascii_uppercase
from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem

our_aloha_information = BlackBoxInformation(
    name="our_aloha",
    max_sequence_length=5,
    aligned=True,
    fixed_length=True,
    deterministic=True,
    alphabet=list(ascii_uppercase),
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)


class OurAlohaBlackBox(AbstractBlackBox):
    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(batch_size, parallelize, num_workers, evaluation_budget)

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        matches = x == np.array(["A", "L", "O", "H", "A"])
        return np.sum(matches, axis=1, keepdims=True)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return our_aloha_information


class OurAlohaProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        # The alphabet: ["A", "B", "C", ...]
        return our_aloha_information

    def create(self, seed: int = None, **kwargs) -> Problem:
        f = OurAlohaBlackBox()
        x0 = np.array([["A", "L", "O", "O", "F"]])

        return Problem(f, x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    aloha_problem_factory = OurAlohaProblemFactory()
    register_problem(
        aloha_problem_factory,
        conda_environment_name="poli_aloha_problem",
    )
