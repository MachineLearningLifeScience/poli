"""
This is a registration script for the toy continuous
objectives which are usually used to benchmark continuous
optimization algorithms in several dimensions.

We focus on the ones that allow us to specify the
dimensionality of the problem [1].

The problem is registered as 'toy_continuous_problem',
and it uses a conda environment called 'poli__base'
(see the environment.yml file in this folder).

The following problems are registered:
TODO: write.
"""
from typing import Tuple
from string import ascii_uppercase

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.seeding import seed_numpy, seed_python

from .toy_continuous_problem import (
    POSSIBLE_FUNCTIONS,
    ToyContinuousProblem,
)


class ToyContinuousBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        function_name: str = None,
        n_dimensions: int = 2,
        embed_in: int = None,
    ):
        self.alphabet = None

        assert (
            function_name in POSSIBLE_FUNCTIONS
        ), f"'{function_name}' is not a valid function name. Expected it to be one of {POSSIBLE_FUNCTIONS}."

        self.function_name = function_name
        self.n_dimensions = n_dimensions
        self.embed_in = embed_in
        self.function = ToyContinuousProblem(
            function_name,
            n_dims=n_dimensions,
            embed_in=embed_in,
        )
        self.bounds = self.function.limits

        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        # x is a [b, L] array of strings or ints, if they are
        # ints, then we should convert them to strings
        # using the alphabet.
        # TODO: this assumes that the input is a batch of size 1.
        # Address this when we change __call__.
        if not x.dtype.kind == "f":
            raise ValueError("Expected a batch of floats. ")

        return self.function(x)


class ToyContinuousProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(
            name="toy_continuous_problem",
            max_sequence_length=np.inf,
            aligned=True,
            alphabet=None,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        function_name: str = None,
        n_dimensions: int = 2,
        embed_in: int = None,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        assert (
            function_name in POSSIBLE_FUNCTIONS
        ), f"'{function_name}' is not a valid function name. Expected it to be one of {POSSIBLE_FUNCTIONS}."

        # We set the seed for numpy and python
        seed_numpy(seed)
        seed_python(seed)

        problem_info = self.get_setup_information()
        f = ToyContinuousBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            function_name=function_name,
            n_dimensions=n_dimensions,
            embed_in=embed_in,
        )
        if embed_in is None:
            x0 = np.array([[0.0] * n_dimensions])
        else:
            x0 = np.array([[0.0] * embed_in])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    toy_continuous_problem_factory = ToyContinuousProblemFactory()
    register_problem(
        toy_continuous_problem_factory,
        conda_environment_name="poli__base",
        # force=True
    )
