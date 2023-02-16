import numpy as np

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper


class ExternalObserver(AbstractObserver):
    def __init__(self, observer_script):
        self.observer_script = observer_script
        self.process_wrapper = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.process_wrapper.send([x, y, context])

    def initialize_observer(self, setup_info: ProblemSetupInformation, caller_info, x0, y0) -> str:
        # start observer process
        self.process_wrapper = ProcessWrapper(self.observer_script)
        # send setup information
        self.process_wrapper.send([setup_info, caller_info, x0, y0])
        # wait for logger handle
        observer_info = self.process_wrapper.recv()
        # forward to objective factory
        return observer_info

    def finish(self) -> None:
        self.process_wrapper.send(None)
        self.process_wrapper.close()
