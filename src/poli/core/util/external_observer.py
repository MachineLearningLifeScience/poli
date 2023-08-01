import numpy as np

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper
from poli.core.registry import config, _DEFAULT, _OBSERVER


class ExternalObserver(AbstractObserver):
    """
    This is an observer class used by poli to wrap observer functionality.
    User-defined observers typically do NOT inherit from here.
    """

    def __init__(self):
        self.observer_script = config[_DEFAULT][_OBSERVER]
        self.process_wrapper = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.process_wrapper.send([x, y, context])

    def initialize_observer(
        self, setup_info: ProblemSetupInformation, caller_info, x0, y0, seed
    ) -> str:
        # start observer process
        # VERY IMPORTANT: the observer script MUST accept port and password as arguments
        self.process_wrapper = ProcessWrapper(self.observer_script)
        # send setup information
        self.process_wrapper.send([setup_info, caller_info, x0, y0, seed])
        # wait for logger handle
        observer_info = self.process_wrapper.recv()
        # forward to objective factory
        return observer_info

    def finish(self) -> None:
        if self.process_wrapper is not None:
            self.process_wrapper.send(None)
            self.process_wrapper.close()
            self.process_wrapper = None
