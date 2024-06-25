"""A simple example of how to log objective function calls.
"""

import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class SimplePrintObserver(AbstractObserver):
    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> None:
        return None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        for x_i, y_i in zip(x.tolist(), y.tolist()):
            print(f"{x_i} -> {y_i}")

    def log(self, algorithm_info: dict):
        print(algorithm_info)

    def finish(self) -> None:
        pass
