from typing import Any
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

    def __init__(self, observer_name: str = None, **kwargs_for_observer):
        if observer_name is None:
            observer_name = _DEFAULT

        self.observer_script = config[observer_name][_OBSERVER]
        self.process_wrapper = None
        self.kwargs_for_observer = kwargs_for_observer

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # We send the observation
        self.process_wrapper.send(["OBSERVATION", x, y, context])

        # And we make sure the process received and logged it correctly
        msg_type, *msg = self.process_wrapper.recv()
        if msg_type == "EXCEPTION":
            e, tb = msg
            print(tb)
            raise e

        # else, it was a successful observation

    def initialize_observer(
        self, setup_info: ProblemSetupInformation, caller_info, x0, y0, seed
    ) -> str:
        # start observer process
        # VERY IMPORTANT: the observer script MUST accept port and password as arguments
        self.process_wrapper = ProcessWrapper(self.observer_script)

        # send setup information
        self.process_wrapper.send(
            [setup_info, caller_info, x0, y0, seed, self.kwargs_for_observer]
        )

        # wait for logger handle
        msg_type, *msg = self.process_wrapper.recv()
        if msg_type == "SETUP":
            observer_info = msg[0]
        elif msg_type == "EXCEPTION":
            e, tb = msg
            print(tb)
            raise e
        else:
            raise ValueError("Unknown message type from observer process: " + msg_type)

        # forward to objective factory
        return observer_info

    def finish(self) -> None:
        if self.process_wrapper is not None:
            self.process_wrapper.send(["QUIT", None])
            self.process_wrapper.close()
            self.process_wrapper = None

    def __getattr__(self, __name: str) -> Any:
        """
        Asks for the attribute of the underlying
        black-box function by sending a message
        to the process w. the msg_type "ATTRIBUTE".
        """
        self.process_wrapper.send(["ATTRIBUTE", __name])
        msg_type, *msg = self.process_wrapper.recv()
        if msg_type == "EXCEPTION":
            e, tb = msg
            print(tb)
            raise e
        else:
            assert msg_type == "ATTRIBUTE"
            attribute = msg[0]
            return attribute
