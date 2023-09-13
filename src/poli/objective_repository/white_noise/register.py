"""
This is a registration script for the white_noise problem,
whose black box objective function returns standard Gaussian
noise.

The problem is registered as 'white_noise', and it uses
a conda environment called 'poli__base' (see the
environment.yml file in this folder).
"""
from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation


class WhiteNoiseBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
    ):
        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        return np.random.randn(x.shape[0], 1)


class WhiteNoiseProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        # A mock alphabet made of the 10 digits.
        alphabet = [str(i) for i in range(10)]

        return ProblemSetupInformation(
            name="white_noise",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=alphabet,
        )

    def create(
        self,
        seed: int = 0,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        problem_info = self.get_setup_information()
        f = WhiteNoiseBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )
        x0 = np.array([["1", "2", "3"]])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    white_noise_problem_factory = WhiteNoiseProblemFactory()
    register_problem(
        white_noise_problem_factory,
        conda_environment_name="poli__base",
        # force=True
    )
