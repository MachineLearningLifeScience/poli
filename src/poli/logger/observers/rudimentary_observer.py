import numpy as np

from src.poli.core.util.abstract_observer import AbstractObserver

BLACKBOX = "blackbox"
SEQUENCE = "sequence"


class RudimentaryObserver(AbstractObserver):
    def __init__(self):
        super().__init__()
        self.step = 0

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        self.logger.log({SEQUENCE: x, BLACKBOX: y[0, 0]}, step=self.step)
        self.step += 1
