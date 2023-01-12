"""
This is the main file relevant for users who want to run objective functions.
"""
import logging
import os
import subprocess
import warnings
from typing import Callable
import numpy as np
from multiprocessing.connection import Listener

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.registry import config, _RUN_SCRIPT_LOCATION, _DEFAULT, _OBSERVER
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.external_observer import ExternalObserver


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
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret password')
    # start objective process
    objective_run_script = config[name][_RUN_SCRIPT_LOCATION]
    proc = subprocess.Popen(objective_run_script, stdout=None, stderr=None, cwd=os.getcwd())
    # TODO: add signal listener that intercepts when proc ends
    # wait for connection from objective process
    # TODO: potential (unlikely) race condition! (process might try to connect before listener is ready!)
    conn = listener.accept()
    # wait for objective process to finish setting up
    x0, y0, problem_information = conn.recv()

    # instantiate observer
    observer = None
    observer_info = None
    observer_script = config[_DEFAULT][_OBSERVER]
    if observer_script != '':
        # try:
            observer: AbstractObserver = ExternalObserver(observer_script)
            observer_info = observer.initialize_observer(problem_information, caller_info, x0, y0)
        # except Exception as e:
        #     warnings.warn("Could not instantiate observer. Exception was: ")
        #     logging.exception(e)

    def f(x: np.ndarray, context=None) -> np.ndarray:
        # send input and wait for reply
        conn.send([x, context])
        val = conn.recv()
        #if isinstance(val, Exception):
        #    raise val
        if observer is not None:
            for i in range(x.shape[0]):
                observer.observe(x[i:i+1, :], val[i:i+1, :], context)
        return val

    def terminate():
        # terminate objective process
        conn.send(None)
        # terminate observer
        observer.finish()
        #listener.close()  # no need to close the connection, the objective does
        #proc.terminate()

    return problem_information, f, x0, y0, observer_info, terminate
