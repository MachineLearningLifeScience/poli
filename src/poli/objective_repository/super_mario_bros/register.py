"""
This script registers Super Mario Bros as an
objective function for POLi.

The metric being recorded is the number of
jumps Mario makes in a given level. The goal
is, of course, maximizing it.
"""
from typing import Tuple
from pathlib import Path

import torch
import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from model import load_example_model

from simulator import test_level_from_z


THIS_DIR = Path(__file__).parent.resolve()

# TODO: download the model from the internet
# if it doesn't exist, as well as the simulator
# FIXME: do this after we remove .pt and .jar
# files from the python installation.


class SMBBlackBox(AbstractBlackBox):
    def __init__(self, L: int):
        super().__init__(L)

        self.model = load_example_model(THIS_DIR / "example.pt")

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Runs the given input x as a latent code
        through the model and returns the number
        of jumps Mario makes in the level.
        """
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
            jumps = -10.0
        return np.array([jumps], dtype=float).reshape(1, 1)


class SMBProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        alphabet_symbols = ["z1", "z2"]
        alphabet = {symbol: i for i, symbol in enumerate(alphabet_symbols)}

        return ProblemSetupInformation(
            name="super_mario_bros",
            max_sequence_length=2,
            aligned=True,
            alphabet=alphabet,
        )

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        L = self.get_setup_information().get_max_sequence_length()
        f = SMBBlackBox(L)
        x0 = np.ones([1, L])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # (once) we have to register our factory
    smb_problem_factory = SMBProblemFactory()
    register_problem(
        smb_problem_factory,
        conda_environment_name="poli__mario",
    )
