"""Registers the Super Mario Bros objective function and factory.

The metric being recorded is the number of
jumps Mario makes in a given level. The goal
is, of course, maximizing it while keeping the
level playable.

"""

from pathlib import Path

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.super_mario_bros.information import SMB_ALPHABET

THIS_DIR = Path(__file__).parent.resolve()

# TODO: download the model from the internet
# if it doesn't exist, as well as the simulator
# FIXME: do this after we remove .pt and .jar
# files from the python installation.


class SuperMarioBrosBlackBox(AbstractBlackBox):
    """
    Black box which returns how many jumps were performed
    by Mario in the given level.

    Parameters
    ----------
    max_time : int, optional
        The maximum time for the simulation in seconds, by default 30.
    visualize : bool, optional
        Flag indicating whether to visualize the simulation, by default False.
    batch_size : int, optional
        The batch size for simultaneous execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Methods
    -------
    _black_box(self, x, context=None)
        Runs the given input x as a latent code
        through the model and returns the number
        of jumps Mario makes in the level. If the
        level is not solvable, returns np.nan.
    """

    def __init__(
        self,
        max_time: int = 30,
        visualize: bool = False,
        value_on_unplayable: float = np.nan,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        """
        Initializes a new instance of the SMBBlackBox class.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for simultaneous execution, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize execution, by default False.
        num_workers : int, optional
            The number of workers for parallel execution, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.force_isolation = force_isolation
        self.max_time = int(max_time)
        self.visualize = visualize
        self.value_on_unplayable = value_on_unplayable
        _ = get_inner_function(
            isolated_function_name="super_mario_bros__isolated",
            class_name="SMBIsolatedLogic",
            module_to_import="poli.objective_repository.super_mario_bros.isolated_function",
            force_isolation=self.force_isolation,
            alphabet=self.get_black_box_info().alphabet,
            max_time=self.max_time,
            visualize=self.visualize,
            value_on_unplayable=self.value_on_unplayable,
        )

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """Computes number of jumps in a given latent code x."""
        inner_function = get_inner_function(
            isolated_function_name="super_mario_bros__isolated",
            class_name="SMBIsolatedLogic",
            module_to_import="poli.objective_repository.super_mario_bros.isolated_function",
            force_isolation=self.force_isolation,
            quiet=True,
            alphabet=self.get_black_box_info().alphabet,
            max_time=self.max_time,
            visualize=self.visualize,
            value_on_unplayable=self.value_on_unplayable,
        )
        return inner_function(x, context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="super_mario_bros",
            max_sequence_length=14 * 14,
            aligned=True,
            fixed_length=True,
            deterministic=False,
            alphabet=SMB_ALPHABET,
            log_transform_recommended=True,
            discrete=True,
            padding_token=None,
        )


class SuperMarioBrosProblemFactory(AbstractProblemFactory):
    """
    Problem factory for the Super Mario Bros objective function.

    Methods
    -------
    create(...) -> Tuple[SMBBlackBox, np.ndarray, np.ndarray]
        Creates a new instance of the SMBBlackBox class.
    """

    def create(
        self,
        max_time: int = 30,
        visualize: bool = False,
        value_on_unplayable: float = np.nan,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """Creates a new instance of the Super Mario Bros problem.

        Parameters
        ----------
        max_time : int, optional
            The maximum time for the simulation in seconds, by default 30.
        visualize : bool, optional
            Flag indicating whether to visualize the simulation, by default False.
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
        force_isolation: bool, optional
            Flag indicating whether to force isolation inside the black box,
            by default False.

        Returns
        -------
        problem : Problem:
            The problem instance containing the black box and an initial point.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = SuperMarioBrosBlackBox(
            max_time=max_time,
            visualize=visualize,
            value_on_unplayable=value_on_unplayable,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        x0 = np.array([["-"] * 14] * 14)
        x0[-1, :] = "X"
        x0 = x0.reshape(1, 14 * 14)

        problem = Problem(f, x0)

        return problem
