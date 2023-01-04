"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import subprocess
from typing import Callable

import numpy as np
from multiprocessing.connection import Listener


def create(name: str, caller_info) -> (Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, str, Callable):
    """
    Instantiantes a black-box function.
    :param name:
        The name of the objective function or a shell-script for execution.
    :param caller_info:
        Optional information about the caller that is forwarded to the logger to initialize the run.
    :return:
        f: an objective function that accepts a numpy array and returns a numpy array
        x0: initial inputs
        y0: f(x0)
        run_info: information from the logger about the instantiated run (allows the calling algorithm to connect)
        terminate: a function to end the process behind f
    """
    cwd = os.getcwd()
    if not name.endswith(".sh"):
        #cwd = os.path.dirname(__file__)  # execute script in project main folder
        name = os.path.join(os.path.dirname(__file__), ".", "objective_run_scripts", name + ".sh")
    proc = subprocess.Popen(name, stdout=None, stderr=None, cwd=cwd)

    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    conn.send(caller_info)
    x0, y0, run_info = conn.recv()

    def f(x: np.ndarray) -> np.ndarray:
        conn.send(x)
        val = conn.recv()
        return val

    def terminate():
        conn.send(None)

    return f, x0, y0, run_info, terminate
