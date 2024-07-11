"""Registers the Super Mario Bros objective function and factory.

The metric being recorded is the number of
jumps Mario makes in a given level. The goal
is, of course, maximizing it while keeping the
level playable.

"""

from pathlib import Path
from typing import List

import numpy as np

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.objective_repository.super_mario_bros.information import smb_info
from poli.objective_repository.super_mario_bros.simulator import (
    test_level_from_int_array,
)

THIS_DIR = Path(__file__).parent.resolve()

# TODO: download the simulator from the internet
# if it doesn't exist
# FIXME: do this after we remove .jar
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
        Runs the given input x as a flattened level (14x14)
        through the model and returns the number
        of jumps Mario makes in the level. If the
        level is not solvable, returns np.nan.
    """

    def __init__(
        self,
        alphabet: List[str] = smb_info.alphabet,
        max_time: int = 30,
        visualize: bool = False,
        value_on_unplayable: float = np.nan,
    ):
        self.alphabet = alphabet
        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        self.max_time = max_time
        self.visualize = visualize
        self.value_on_unplayable = value_on_unplayable

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        """Computes number of jumps in a given latent code x."""
        assert x.ndim == 2, "x must be a 2D array"
        batch_size, _ = x.shape

        # Converting the level to ints
        x = np.array(
            [[self.alphabet_s_to_i[x_ij] for x_ij in x_i] for x_i in x]
        ).reshape(batch_size, 14, 14)

        # Run the model
        jumps_for_all_levels = []
        for level in x:
            res = test_level_from_int_array(
                level, max_time=self.max_time, visualize=self.visualize
            )

            if not isinstance(res, dict):
                raise ValueError(
                    "Something probably went wrong with the Java simulation "
                    "of the level. It is quite likely you haven't set up a "
                    "virtual screen/frame buffer. Check the docs."
                )

            # Return the number of jumps if the level was
            # solved successfully, else return np.nan
            if res["marioStatus"] == 1:
                jumps = res["jumpActionsPerformed"]
            else:
                jumps = self.value_on_unplayable

            jumps_for_all_levels.append(jumps)

        return np.array(jumps_for_all_levels, dtype=float).reshape(batch_size, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        SMBIsolatedLogic,
        name="super_mario_bros__isolated",
        conda_environment_name="poli__mario",
    )
