from typing import Any

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper


class ExternalBlackBox(AbstractBlackBox):
    """An external version of the black-box function to be instantiated in isolated processes."""

    def __init__(self, process_wrapper: ProcessWrapper):
        """
        Initialize the ExternalBlackBox object.

        Parameters
        ----------
        info : ProblemSetupInformation
            The information about the problem.
        process_wrapper : ProcessWrapper
            The process wrapper to communicate with the objective process.
        """
        # We don't need to pass the kwargs to the parent class,
        # because we overwrite the __getattr__ method to communicate
        # with the isolated objective process.
        super().__init__()
        self.process_wrapper = process_wrapper

    def _black_box(self, x, context=None):
        """
        Evaluates the black-box function.

        In this external black-box, the evaluation is done by sending a message
        to the objective process and waiting for the response. The interprocess
        communication is handled by the ProcessWrapper class, which maintains
        an isolated process in which a black-box function runs with, potentially,
        a completely different python executable (e.g. inside another conda
        environment).

        Parameters
        ----------
        x : np.ndarray
            The input data points.
        context : object
            Additional context for the observation.

        Returns
        -------
        y : np.ndarray
            The output data points.
        """
        self.process_wrapper.send(["QUERY", x, context])
        msg_type, *val = self.process_wrapper.recv()
        if msg_type == "EXCEPTION":
            e, traceback_ = val
            print(traceback_)
            raise e
        elif msg_type == "QUERY":
            y = val[0]

            return y
        else:
            raise ValueError(
                f"Internal error: received {msg_type} when expecting QUERY or EXCEPTION"
            )

    def terminate(self):
        """Terminates the external black box."""
        # terminate objective process
        if self.process_wrapper is not None:
            try:
                self.process_wrapper.send(["QUIT", None])
                self.process_wrapper.close()  # clean up connection
            except AttributeError:
                # This means that the process has already been terminated
                pass
            self.process_wrapper = None
        # terminate observer
        if self.observer is not None:
            try:
                self.observer.finish()
                self.observer = None
            except:
                pass

    def __getattr__(self, __name: str) -> Any:
        """Gets an attribute from the underlying black-box function.

        Asks for the attribute of the underlying
        black-box function by sending a message
        to the process w. the msg_type "ATTRIBUTE".

        Parameters
        ----------
        __name : str
            The name of the attribute.

        Returns
        -------
        attribute : Any
            The attribute of the underlying black-box function.
        """
        self.process_wrapper.send(["ATTRIBUTE", __name])
        msg_type, *msg = self.process_wrapper.recv()
        if msg_type == "EXCEPTION":
            e, traceback_ = msg
            print(traceback_)
            raise e
        else:
            assert msg_type == "ATTRIBUTE"
            attribute = msg[0]
            return attribute

    def __del__(self):
        self.terminate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.terminate()
