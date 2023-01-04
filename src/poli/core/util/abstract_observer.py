import numpy as np

from src.poli.core.util.abstract_logger import AbstractLogger


class AbstractObserver:
    def __init__(self):
        self.logger = None

    def initialize(self, logger: AbstractLogger) -> None:
        """
        A separate initialization function to allow parameter-free instantiation.
        :param logger:
        """
        self.logger = logger

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("abstract method")
