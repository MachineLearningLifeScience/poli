__author__ = 'Simon Bartels'

import os.path

import numpy as np

import poli.core.registry
from poli.core.AbstractBlackBox import BlackBox
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.AbstractProblemFactory import AbstractProblemFactory

AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]: i for i in range(len(AA))}


class WhiteNoiseFactory(AbstractProblemFactory):
    """
    This is a simple test problem, mainly for debugging purposes but also to test how algorithms behave on problems
    where they are NOT supposed to perform.
    """
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(name="WHITE_NOISE", max_sequence_length=5, aligned=True, alphabet=AA_IDX)

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        class WhiteNoiseBlackBox(BlackBox):
            def _black_box(self, x: np.ndarray) -> np.ndarray:
                # the returned value must be a [1, 1] array
                return np.random.randn(1).reshape(-1, 1)

        f = WhiteNoiseBlackBox(self.get_setup_information().get_max_sequence_length())
        x = np.diag(np.arange(11) + 6) @ np.ones([11, self.get_setup_information().get_max_sequence_length()], dtype=np.int)
        y = f(x)
        return f, x, y


if __name__ == '__main__':
    poli.core.registry.register_problem(WhiteNoiseFactory().get_setup_information().get_problem_name(),
                                        os.path.join(os.path.dirname(__file__), "white_noise.sh"))
