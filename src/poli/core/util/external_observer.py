"""External observer, which can be run in an isolated process."""

from typing import Any

import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.registry import _DEFAULT, _OBSERVER, config
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper


class ExternalObserver(AbstractObserver):
    """An external version of the observer class to be instantiated in isolated processes.

    Parameters
    ----------
    observer_name : str, optional
        The name of the observer. If not provided, the default observer name will be used.
    **kwargs_for_observer
        Additional keyword arguments to be passed to the observer.

    Methods
    -------
    observe(x, y, context=None)
        Sends the observation to the observer process and verifies if it was logged correctly.
    initialize_observer(setup_info, caller_info, x0, y0, seed)
        Starts the observer process and sends the setup information.
    finish()
        Closes the observer process.
    __getattr__(__name)
        Retrieves the attribute of the underlying observer.
    """

    def __init__(self, observer_name: str = None, **kwargs_for_observer):
        """
        Initialize the ExternalObserver object.

        Parameters
        ----------
        observer_name : str, optional
            The name of the observer. If not provided, the default observer will be used.
        **kwargs_for_observer
            Additional keyword arguments to be passed to the observer.
        """
        if observer_name is None:
            observer_name = _DEFAULT

        self.observer_name = observer_name

        self.observer_script = config[_OBSERVER][observer_name]
        self.process_wrapper = None
        self.kwargs_for_observer = kwargs_for_observer

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        """
        Observe the given data points.

        Parameters
        ----------
        x: np.ndarray
            The input data points.
        y: np.ndarray
            The output data points.
        context: object
            Additional context for the observation.

        Raises
        -------
        Exception:
            If the underlying observer process raises an exception.
        """

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
        self, setup_info: BlackBoxInformation, caller_info, seed
    ) -> str:
        """
        Initialize the observer.

        Parameters
        ----------
        problem_setup_info : ProblemSetupInformation
            The information about the problem setup.
        caller_info : object
            The information about the caller.
        seed : int
            The seed value for random number generation.

        Returns
        -------
        observer_info: object
            Relevant information returned by the underlying observer's
            initialize_observer method.

        Raises
        ------
        Exception:
            Any exception raised by the underlying observer.
        ValueError:
            If the message type received from the observer process is unknown.
        """
        # start observer process
        # VERY IMPORTANT: the observer script MUST accept port and password as arguments
        self.process_wrapper = ProcessWrapper(self.observer_script)

        # send setup information
        self.process_wrapper.send(
            [setup_info, caller_info, seed, self.kwargs_for_observer]
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

    def log(self, algorithm_info: dict):
        # We send the observation
        self.process_wrapper.send(["LOG", algorithm_info])

        # And we make sure the process received and logged it correctly
        msg_type, *msg = self.process_wrapper.recv()
        if msg_type == "EXCEPTION":
            e, tb = msg
            print(tb)
            raise e

        # else, it was a successful observation

    def finish(self) -> None:
        """Finish the external observer process.

        This method sends a "QUIT" message to the process wrapper and closes it.
        """
        if self.process_wrapper is not None:
            self.process_wrapper.send(["QUIT", None])
            self.process_wrapper.close()
            self.process_wrapper = None

    def __getattr__(self, __name: str) -> Any:
        """Get an attribute of the underlying observer if it exists.

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.observer_name})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.observer_name}, script_location={self.observer_script})>"
