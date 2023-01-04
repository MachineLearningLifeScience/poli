__author__ = 'Simon Bartels'
import numpy as np

from core.AbstractBlackBox import BlackBox
from core.AbstractProblemFactory import ProblemSetupInformation
from objectives.lambo_foldx.abstract_foldx_factory import AbstractFoldXFactory

AA = ['a', 'r', 'n', 'd']
AA_IDX = {AA[i]: i for i in range(len(AA))}


class WhiteNoiseFactory(AbstractFoldXFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(name="WHITE_NOISE", max_sequence_length=5, aligned=True, alphabet=AA_IDX)

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        class WhiteNoiseBlackBox(BlackBox):
            def _black_box(self, x: np.ndarray) -> np.ndarray:
                # the returned value must be a [1, 1] array
                return np.random.randn(1).reshape(-1, 1)

        f = WhiteNoiseBlackBox(self.get_setup_information().get_max_sequence_length())
        x = np.zeros([1, self.get_setup_information().get_max_sequence_length()])
        y = f(x)
        return f, x, y
