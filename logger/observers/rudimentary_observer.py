import numpy as np

from core.util.abstract_logger import AbstractLogger
from core.util.abstract_observer import AbstractObserver


class RudimentaryObserver(AbstractObserver):
    def __init__(self, logger: AbstractLogger):
        self.logger = logger
        self.step = 0

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        self.logger.log({SEQUENCE: x, BLACKBOX: y[0, 0]}, step=self.step)
        self.step += 1
