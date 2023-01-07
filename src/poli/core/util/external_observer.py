import os
import subprocess
import numpy as np
from multiprocessing.connection import Listener

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver


class ExternalObserver(AbstractObserver):
    def __init__(self, observer_script):
        self.observer_script = observer_script
        self.conn = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.conn.send([x, y, context])

    def initialize_observer(self, setup_info: ProblemSetupInformation, caller_info) -> str:
        cwd = os.getcwd()
        proc = subprocess.Popen(self.observer_script, stdout=None, stderr=None, cwd=cwd)

        address = ('localhost', 6001)
        listener = Listener(address, authkey=b'secret password')
        self.conn = listener.accept()
        self.conn.send([setup_info, caller_info])
        observer_info = self.conn.recv()
        return observer_info

    def finish(self) -> None:
        self.conn.send(None)
