"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import signal
import subprocess
import numpy as np
from multiprocessing.connection import Listener

from core.registry import INIT_DATA_FILE, INPUT_DATA_FILE, OUTPUT_DATA_FILE


def create(name: str, caller_info):
    cwd = os.path.dirname(__file__)  # execute script in project main folder
    # write caller_info for logger in initialization
    proc = subprocess.Popen("./objective_run_scripts/" + name + ".sh", stdout=None, stderr=None, cwd=cwd)

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
        conn.close()

    return f, x0, y0, run_info, terminate
