import numpy as np

from core.AbstractBlackBox import BlackBox
from core.problem_setup_information import ProblemSetupInformation


class AbstractProblemFactory:
    def get_setup_information(self) -> ProblemSetupInformation:
        raise NotImplementedError("abstract method")

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        """
        Returns a blackbox function and initial observations.
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")
