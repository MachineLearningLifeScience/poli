__author__ = "Simon Bartels"
import numpy as np
import logging

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver


class MyObserver(AbstractObserver):
    def __init__(self):
        self.step = 1

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        logging.fatal(f"observer has been called in step {self.step}: f({x})={y}")
        self.step += 1

    def initialize_observer(
        self,
        problem_setup_info: ProblemSetupInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
    ) -> object:
        return None

    def finish(self) -> None:
        pass


if __name__ == "__main__":
    from poli import objective_factory
    from poli.core.registry import set_observer

    # (once) we have to register our observer
    set_observer(MyObserver(), conda_environment_location="")

    problem_info, f, x0, y0, run_info = objective_factory.create(
        "MY_PROBLEM", observer_init_info=None
    )
    # call objective function and observe that observer is called
    print(f"The observer will be called {x0.shape[0]} time(s).")
    f(x0)
    f.terminate()
