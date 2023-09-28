import numpy as np
from poli.core.problem_setup_information import ProblemSetupInformation


class AbstractObserver:
    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        raise NotImplementedError("abstract method")

    def initialize_observer(
        self,
        problem_setup_info: ProblemSetupInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
    ) -> object:
        raise NotImplementedError("abstract method")

    def finish(self) -> None:
        pass

    def __del__(self):
        # TODO: this is not pretty. We should find a better
        # way to detect when the observer process has been
        # terminated.
        try:
            self.finish()
        except Exception:
            # This means that the observer process has already been terminated.
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
