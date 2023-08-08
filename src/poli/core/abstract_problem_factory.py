from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation


class AbstractProblemFactory:
    def get_setup_information(self) -> ProblemSetupInformation:
        raise NotImplementedError("abstract method")

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Returns a blackbox function and initial observations.
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")
