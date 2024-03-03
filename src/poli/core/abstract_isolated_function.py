import numpy as np


class AbstractIsolatedFunction:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        raise NotImplementedError
