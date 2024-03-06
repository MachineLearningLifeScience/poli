"""Registers the Super Mario Bros objective function and factory.

The metric being recorded is the number of
jumps Mario makes in a given level. The goal
is, of course, maximizing it while keeping the
level playable.

"""

from typing import Tuple
from pathlib import Path

import torch
import numpy as np

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.seeding import seed_python_numpy_and_torch

from poli.objective_repository.super_mario_bros.model import load_example_model

from poli.objective_repository.super_mario_bros.simulator import test_level_from_z


THIS_DIR = Path(__file__).parent.resolve()

# TODO: download the model from the internet
# if it doesn't exist, as well as the simulator
# FIXME: do this after we remove .pt and .jar
# files from the python installation.


class SMBIsolatedLogic(AbstractIsolatedFunction):
    """
    Black box which returns how many jumps were performed
    by Mario in the given level.

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

    Attributes
    ----------
    model : torch.nn.Module
        The model used for the simulation.

    Methods
    -------
    _black_box(self, x, context=None)
        Runs the given input x as a latent code
        through the model and returns the number
        of jumps Mario makes in the level. If the
        level is not solvable, returns np.NaN.
    """

    def __init__(self):
        """
        Initializes a new instance of the SMBBlackBox class.
        """
        self.model = load_example_model(THIS_DIR / "example.pt")

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        """Computes number of jumps in a given latent code x."""
        z = torch.from_numpy(x).float()
        z = z.unsqueeze(0)

        # Run the model
        with torch.no_grad():
            res = test_level_from_z(z, self.model, visualize=True)

        # Return the number of jumps if the level was
        # solved successfully, else return np.NaN
        if res["marioStatus"] == 1:
            jumps = res["jumpActionsPerformed"]
        else:
            jumps = np.nan

        return np.array([jumps], dtype=float).reshape(1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        SMBIsolatedLogic,
        name="super_mario_bros__isolated",
        conda_environment_name="poli__mario",
    )
