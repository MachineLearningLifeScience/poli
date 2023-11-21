"""
This script implements Super Mario Bros as an
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


class SMBBlackBox(AbstractBlackBox):
    def __init__(self, info: ProblemSetupInformation, batch_size: int = None):
        super().__init__(info=info, batch_size=batch_size)

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

        # Return the number of jumps
        return np.array([res["jumpActionsPerformed"]], dtype=float).reshape(1, 1)


class SMBProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        alphabet = ["z1", "z2"]

        return ProblemSetupInformation(
            name="SMB",
            max_sequence_length=2,
            aligned=True,
            alphabet=alphabet,
        )

    def create(
        self, seed: int = None
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        problem_info = self.get_setup_information()
        f = SMBBlackBox(info=problem_info)
        sequence_length = problem_info.get_max_sequence_length()
        x0 = np.ones([1, sequence_length])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli import objective_factory
    from poli.core.registry import register_problem

    # (once) we have to register our factory
    smb_problem_factory = SMBProblemFactory()
    register_problem(
        smb_problem_factory,
        conda_environment_name="poli__mario",
    )

    # now we can instantiate our objective
    problem_name = smb_problem_factory.get_setup_information().get_problem_name()
    problem_info, f, x0, y0, run_info = objective_factory.create(
        problem_name, caller_info=None, observer=None
    )

    print(f(x0[:1, :]))
    f.terminate()
