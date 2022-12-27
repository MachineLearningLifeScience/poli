import numpy as np


class AbstractObserver:
    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("abstract method")
