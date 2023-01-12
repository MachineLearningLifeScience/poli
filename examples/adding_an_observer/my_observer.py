__author__ = 'Simon Bartels'
import os
import numpy as np
import logging

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.observer_wrapper import start_observer_process


class MyObserver(AbstractObserver):
    def __init__(self):
        self.step = 1

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        logging.info(f"step {self.step}: f({x})={y}")
        self.step += 1

    def initialize_observer(self, problem_setup_info: ProblemSetupInformation, caller_info: object, x0: np.ndarray, y0: np.ndarray) -> object:
        return None

    def finish(self) -> None:
        pass


if __name__ == '__main__':
    observer_name = os.path.basename(__file__)[:-2] + MyObserver.__name__
    start_observer_process(observer_name)
