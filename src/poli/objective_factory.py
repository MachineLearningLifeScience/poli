"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import subprocess
from typing import Callable
import numpy as np
from multiprocessing.connection import Listener

from poli.core.problem_setup_information import ProblemSetupInformation


# TODO: typing information about f out-dated? Would be nice to replace this by a class
def create(name: str, caller_info) -> (ProblemSetupInformation, Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, str, Callable):
    """
    Instantiantes a black-box function.
    :param name:
        The name of the objective function or a shell-script for execution.
    :param caller_info:
        Optional information about the caller that is forwarded to the logger to initialize the run.
    :return:
        problem_information: a ProblemSetupInformation object holding basic properties about the problem
        f: an objective function that accepts a numpy array and returns a numpy array
        x0: initial inputs
        y0: f(x0)
        observer_info: information from the observer_info about the instantiated run (allows the calling algorithm to connect)
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
    x0, y0, problem_information, observer_info = conn.recv()

    def f(x: np.ndarray, context=None) -> np.ndarray:
        conn.send([x, context])
        val = conn.recv()
        return val

    def terminate():
        conn.send(None)

    return problem_information, f, x0, y0, observer_info, terminate
