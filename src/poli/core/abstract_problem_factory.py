from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation


class MetaProblemFactory(type):
    def __repr__(cls) -> str:
        try:
            problem_info = cls().get_setup_information()
        except NotImplementedError:
            return f"<{cls.__name__}()>"

        return f"<{cls.__name__}(L={problem_info.max_sequence_length})>"

    def __str__(cls) -> str:
        return f"{cls.__name__}"


class AbstractProblemFactory(metaclass=MetaProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        raise NotImplementedError("abstract method")

    def create(
        self, seed: int = 0, batch_size: int = None
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Returns a blackbox function and initial observations.
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")
