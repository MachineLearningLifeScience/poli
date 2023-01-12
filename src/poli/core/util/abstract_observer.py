import numpy as np
from poli.core.problem_setup_information import ProblemSetupInformation


class AbstractObserver:
    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        raise NotImplementedError("abstract method")

    def initialize_observer(self, problem_setup_info: ProblemSetupInformation, caller_info: object, x0: np.ndarray, y0: np.ndarray) -> object:
        raise NotImplementedError("abstract method")

    def finish(self) -> None:
        raise NotImplementedError("abstract method")
