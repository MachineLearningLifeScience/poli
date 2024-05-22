from __future__ import annotations

import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class MultiObserver(AbstractObserver):
    def __init__(self, observers: list[AbstractObserver]):
        self.observers = observers

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        for observer in self.observers:
            observer.observe(x, y, context)

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
    ) -> object:
        for observer in self.observers:
            observer.initialize_observer(problem_setup_info, caller_info, x0, y0, seed)
