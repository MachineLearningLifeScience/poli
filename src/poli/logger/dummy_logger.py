__author__ = 'Simon Bartels'

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_logger import AbstractLogger


class DummyLogger(AbstractLogger):
    """
    A logger that does nothing. Useful for example for debugging and minimal working examples.
    """
    def log(self, metrics: dict, step: int) -> None:
        pass

    def initialize_logger(self, problem_setup_info: ProblemSetupInformation, caller_info) -> str:
        pass

    def finish(self) -> None:
        pass
