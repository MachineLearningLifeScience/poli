import os
import subprocess
import numpy as np

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.ipc import instantiate_listener


class ExternalObserver(AbstractObserver):
    def __init__(self, observer_script):
        self.observer_script = observer_script
        self.conn = None
        self.listener = None
        self.proc = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.conn.send([x, y, context])

    def initialize_observer(self, setup_info: ProblemSetupInformation, caller_info, x0, y0) -> str:
        listener, port, password = instantiate_listener()
        self.listener = listener
        # start observer process
        self.proc = subprocess.Popen([self.observer_script, port, password], stdout=None, stderr=None, cwd=os.getcwd())
        # wait for connection
        # TODO: potential (unlikely) race condition! (process might try to connect before listener is ready!)
        self.conn = self.listener.accept()
        # send setup information
        self.conn.send([setup_info, caller_info, x0, y0])
        # wait for logger handle
        observer_info = self.conn.recv()
        # forward to objective factory
        return observer_info

    def finish(self) -> None:
        self.conn.send(None)
        self.proc.wait()  # wait for observer to finish
        self.listener.close()
