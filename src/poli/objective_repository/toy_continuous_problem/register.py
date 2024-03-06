"""Registers the toy continuous problem objective function and factory.

This is a registration script for the toy continuous
objectives which are usually used to benchmark continuous
optimization algorithms in several dimensions.

We focus on the ones that allow us to specify the
dimensionality of the problem [1].

The problem is registered as 'toy_continuous_problem',
and it uses a conda environment called 'poli__base'
(see the environment.yml file in this folder).
"""

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem

from poli.core.util.seeding import seed_python_numpy_and_torch

from .toy_continuous_problem import (
    POSSIBLE_FUNCTIONS,
    ToyContinuousProblem,
)

from poli.objective_repository.toy_continuous_problem.information import (
    toy_continuous_info,
)


class ToyContinuousBlackBox(AbstractBlackBox):
    """
    A black box implementation for evaluating the Toy Continuous Problem.

    Parameters
    ----------
    batch_size : int, optional
        The batch size for parallel evaluation, by default None.
    parallelize : bool, optional
        Whether to parallelize the evaluation, by default False.
    num_workers : int, optional
        The number of workers for parallel evaluation, by default None.
    evaluation_budget : int, optional
        The maximum number of evaluations, by default float("inf").
    function_name : str
        The name of the toy continuous function to evaluate, by default None.
    n_dimensions : int
        The number of dimensions for the toy continuous function, by default 2.
    embed_in : int, optional
        If not None, the continuous problem is randomly embedded in this dimension.
        By default, None.

    Attributes
    ----------
    function_name : str
        The name of the toy continuous function.
    n_dimensions : int
        The number of dimensions for the toy continuous function.
    embed_in : int
        The dimension in which to embed the problem.
    function : ToyContinuousProblem
        The toy continuous problem instance.
    bounds : Tuple[np.ndarray, np.ndarray]
        The lower and upper bounds for the toy continuous problem.

    Methods
    -------
    _black_box(x, context=None)
        Evaluates the toy continuous problem on a continuous input x.

    """

    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        function_name: str = None,
        n_dimensions: int = 2,
        embed_in: int = None,
    ):

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
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Evaluates the toy continuous problem on a continuous input x.

        Parameters
        ----------
        x : np.ndarray
            The input to evaluate (expected to be an array [b, L] of floats).
        context : dict, optional
            The context of the evaluation, by default None.

        Returns
        -------
        y : np.ndarray
            The evaluation of the toy continuous problem.

        Raises
        ------
        ValueError
            If the input x is not of type float.

        """
        if not x.dtype.kind == "f":
            raise ValueError("Expected a batch of floats. ")

        return self.function(x)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return toy_continuous_info


class ToyContinuousProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        """
        Returns the setup information for the problem.

        Returns
        -------
        problem_info : ProblemSetupInformation
            The setup information for the problem.
        """
        return toy_continuous_info

    def create(
        self,
        function_name: str,
        n_dimensions: int = 2,
        embed_in: int = None,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Problem:
        """
        Creates a new instance of the toy continuous problem.

        Parameters
        ----------
        function_name : str
            The name of the toy continuous function to evaluate.
        n_dimensions : int, optional
            The number of dimensions for the toy continuous function, by default 2.
        embed_in : int, optional
            If not None, the continuous problem is randomly embedded in this dimension.
            By default, None.
        seed : int, optional
            The seed for the random number generator, by default None.
        batch_size : int, optional
            The batch size for simultaneous execution, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize execution, by default False.
        num_workers : int, optional
            The number of workers for parallel execution, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem : Problem
            The problem instance containing the black box and an initial point.

        Raises
        ------
        ValueError
            If the function_name is not one of the valid functions.
        """
        assert (
            function_name in POSSIBLE_FUNCTIONS
        ), f"'{function_name}' is not a valid function name. Expected it to be one of {POSSIBLE_FUNCTIONS}."

        # We set the seed for numpy and python
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = ToyContinuousBlackBox(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            function_name=function_name,
            n_dimensions=n_dimensions,
            embed_in=embed_in,
        )
        # TODO: initial value should maybe vary according to the function.
        if embed_in is None:
            x0 = np.array([[0.0] * n_dimensions])
        else:
            x0 = np.array([[0.0] * embed_in])

        return Problem(f, x0)


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
