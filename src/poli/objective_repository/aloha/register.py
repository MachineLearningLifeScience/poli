"""Aloha objective factory and function.

This is a registration script for the ALOHA problem,
a simple example of a discrete black box objective
function where the goal is to find the sequence
["A", "L", "O", "H", "A"] among all 5-letter sequences.

The problem is registered as 'aloha', and it uses
a conda environment called 'poli__base' (see the
environment.yml file in this folder).
"""

from typing import Literal, Tuple
from string import ascii_uppercase

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem

from poli.core.util.seeding import seed_python_numpy_and_torch


class AlohaBlackBox(AbstractBlackBox):
    """
    Black box implementation for the Aloha problem.

    The aloha problem is a simple discrete black box problem
    where the goal is to find the sequence ["A", "L", "O", "H", "A"]
    among all 5-letter sequences.

    Parameters
    ----------
    batch_size : int, optional
        The batch size for processing multiple inputs simultaneously, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation, by default False.
    num_workers : int, optional
        The number of workers to use for parallel computation, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Attributes
    ----------
    alphabet : dict
        The mapping of symbols to their corresponding indices in the alphabet.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the distance between the 5-letter sequence in
        x and the target sequence ["A", "L", "O", "H", "A"].
    """

    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        """
        Initialize the aloha black box object.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for processing data, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the processing, by default False.
        num_workers : int, optional
            The number of workers to use for parallel processing, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return BlackBoxInformation(
            name="aloha",
            max_sequence_length=5,
            aligned=True,
            fixed_length=True,
            deterministic=True,
            alphabet=list(ascii_uppercase),
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Compute the distance of x to the sequence "ALOHA".

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, 5] containing strings or integers.
            If the elements are integers, they are converted to strings using the alphabet.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y: np.ndarray
            Array of shape [b, 1] containing the distances to the sequence "ALOHA".
        """
        assert x.dtype.kind == "U", "The input must be an array of strings."

        if x.shape[1] == 1:
            assert (
                len(set([len(x_i) for x_i in x])) == 1
            ), "All strings must have the same length."
            x = np.array([list(x_i[0]) for x_i in x])

        matches = x == np.array(["A", "L", "O", "H", "A"])
        values = np.sum(matches.reshape(x.shape[0], 5), axis=1, keepdims=True).reshape(
            x.shape[0], 1
        )
        return values


class AlohaProblem(Problem):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        fidelity: Literal["high", "low"] = "high",
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(
            black_box=black_box,
            x0=x0,
            fidelity=fidelity,
            evaluation_budget=evaluation_budget,
        )


class AlohaProblemFactory(AbstractProblemFactory):
    """
    Factory for the Aloha problem.

    The aloha problem is a simple discrete black box problem
    where the goal is to find the sequence ["A", "L", "O", "H", "A"]
    among all 5-letter sequences.

    Methods
    -------
    get_setup_information()
        Returns the setup information for the problem.

    create(...)
        Creates a problem instance with the specified parameters.
    """

    @staticmethod
    def get_setup_information() -> ProblemSetupInformation:
        """
        Returns the setup information for the problem.

        Returns
        --------
        problem_info: ProblemSetupInformation
            The setup information for the problem.
        """
        # The alphabet: ["A", "B", "C", ...]
        alphabet = list(ascii_uppercase)

        return ProblemSetupInformation(
            name="aloha",
            max_sequence_length=5,
            aligned=True,
            alphabet=alphabet,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Returns an Aloha blackbox function and initial observations.

        Parameters
        -----------
        seed:  int, optional
            The seed for random number generation. Default is None.
        batch_size:  int, optional
            The batch size for parallel evaluation. Default is None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation. Default is False.
        num_workers:  int, optional
            The number of workers for parallel evaluation. Default is None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        --------
        results: Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
            A tuple containing the blackbox function, initial observations
            for input variables, and initial observations for output
            variables.
        """
        # We set the seed for numpy and python
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        problem_info = self.get_setup_information()
        f = AlohaBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
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
