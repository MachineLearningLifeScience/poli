"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import subprocess
from typing import Callable
import numpy as np
from multiprocessing.connection import Listener

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.registry import config, _RUN_SCRIPT_LOCATION


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
    # start objective process
    objective_run_script = config[name][_RUN_SCRIPT_LOCATION]
    proc = subprocess.Popen(objective_run_script, stdout=None, stderr=None, cwd=os.getcwd())
    # TODO: add signal listener that intercepts when proc ends
    # wait for connection from objective process
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    # send caller_info
    conn.send(caller_info)
    # wait for objective process to finish setting up
    x0, y0, problem_information, observer_info = conn.recv()

    def f(x: np.ndarray, context=None) -> np.ndarray:
        # send input and wait for reply
        conn.send([x, context])
        val = conn.recv()
        if isinstance(val, Exception):
            raise val
        return val

    def terminate():
        # terminate objective process
        conn.send(None)
        #listener.close()
        #proc.terminate()

    return problem_information, f, x0, y0, observer_info, terminate
