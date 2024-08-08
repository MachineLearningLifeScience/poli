"""Registers the white noise problem factory and black box.

This is a registration script for the white_noise problem,
whose black box objective function returns standard Gaussian
noise.

The problem is registered as 'white_noise', and it uses
a conda environment called 'poli__base' (see the
environment.yml file in this folder).
"""

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.seeding import seed_python_numpy_and_torch


class WhiteNoiseBlackBox(AbstractBlackBox):
    """
    A toy black box function that generates standard Gaussian noise.

    Parameters
    ----------
    batch_size : int, optional
        The batch size for vectorized evaluation.
    parallelize : bool, optional
        Whether to parallelize the evaluation.
    num_workers : int, optional
        The number of workers for parallel evaluation.
    evaluation_budget : int, optional
        The maximum number of evaluations.

    Methods
    -------
    _black_box(x, context=None)
        Returns standard Gaussian noise.

    """

    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        """
        Initializes a WhiteNoiseBlackBox.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information.
        batch_size : int, optional
            The batch size for vectorized evaluation, by default None (i.e.
            all of the input).
        parallelize : bool, optional
            Whether to parallelize the evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None (which
            corresponds to half the CPUs available, rounded downwards).
        evaluation_budget : int, optional
            The maximum number of evaluations, by default float("inf").
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """Returns standard Gaussian noise.

        Parameters
        ----------
        x : np.ndarray
            The input.
        context : dict, optional
            The context, by default None.

        Returns
        -------
        y : np.ndarray
            Standard Gaussian noise.
        """
        return np.random.randn(x.shape[0], 1)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="white_noise",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            deterministic=False,
            alphabet=[str(i) for i in range(10)],
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class WhiteNoiseProblemFactory(AbstractProblemFactory):
    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Create a white noise problem with the specified parameters.

        Parameters:
        ----------
        seed : int, optional
            The seed value for random number generation. If not provided, no seeding will be performed.
        batch_size : int, optional
            The number of samples to evaluate each time. If not provided, the default batch size will be used.
        parallelize : bool, optional
            Whether to parallelize the evaluation of samples. Defaults to False.
        num_workers : int, optional
            The number of worker processes to use for parallel evaluation. If not provided,
            half the available CPUs will be used (rounded down).
        evaluation_budget : int, optional
            The maximum number of evaluations allowed. Defaults to infinity.

        Returns:
        -------
        problem: WhiteNoiseProblem
            A white noise problem with the specified parameters.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = WhiteNoiseBlackBox(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        x0 = np.array([["1", "2", "3"]])

        white_noise_problem = Problem(black_box=f, x0=x0)

        return white_noise_problem
