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

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.seeding import seed_python_numpy_and_torch

from model import load_example_model

from simulator import test_level_from_z


THIS_DIR = Path(__file__).parent.resolve()

# TODO: download the model from the internet
# if it doesn't exist, as well as the simulator
# FIXME: do this after we remove .pt and .jar
# files from the python installation.


class SMBBlackBox(AbstractBlackBox):
    """
    Black box which returns how many jumps were performed
    by Mario in the given level.

    Parameters
    ----------
    info : ProblemSetupInformation
        The problem setup information.
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

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        """
        Initializes a new instance of the SMBBlackBox class.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information.
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
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.model = load_example_model(THIS_DIR / "example.pt")

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
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


class SMBProblemFactory(AbstractProblemFactory):
    """
    Problem factory for the Super Mario Bros objective function.

    Methods
    -------
    get_setup_information(self)
        Returns the setup information for the problem.
    create(...) -> Tuple[SMBBlackBox, np.ndarray, np.ndarray]
        Creates a new instance of the SMBBlackBox class.
    """

    def get_setup_information(self) -> ProblemSetupInformation:
        """Returns the setup information for the problem."""
        alphabet_symbols = ["z1", "z2"]
        alphabet = {symbol: i for i, symbol in enumerate(alphabet_symbols)}

        return ProblemSetupInformation(
            name="super_mario_bros",
            max_sequence_length=2,
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
    ) -> Tuple[SMBBlackBox, np.ndarray, np.ndarray]:
        """Creates a new instance of the Super Mario Bros problem.

        Parameters
        ----------
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
        f: SMBBlackBox
            The SMBBlackBox instance.
        x0: np.ndarray
            The initial latent code.
        y0: np.ndarray
            The initial objective function value (i.e. number of jumps).
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        info = self.get_setup_information()
        f = SMBBlackBox(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        x0 = np.ones([1, info.max_sequence_length])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # (once) we have to register our factory
    smb_problem_factory = SMBProblemFactory()
    register_problem(
        smb_problem_factory,
        conda_environment_name="poli__mario",
    )
