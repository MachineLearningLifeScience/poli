"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import signal
import subprocess
import numpy as np

from core.registry import INIT_DATA_FILE, INPUT_DATA_FILE, OUTPUT_DATA_FILE


def create(name: str, caller_info):
    cwd = os.path.dirname(__file__)  # execute script in project main folder
    # write caller_info for logger in initialization
    proc = subprocess.Popen("./objective_run_scripts/" + name + ".sh", stdout=None, stderr=None, cwd=cwd, shell=False)

    # TODO: pass caller_info
    print("caller info: " + str(caller_info))

    _wait()
    xy = np.load(INIT_DATA_FILE)
    # TODO: delete file?
    x0 = xy[:, :-1]
    y0 = xy[:, -1:]

    # TODO: load logger handle
    run_info = None

    def f(x: np.ndarray) -> np.ndarray:
        # write x to destination
        np.save(INPUT_DATA_FILE, x)
        # signal process
        proc.send_signal(signal.SIG_BLOCK)
        _wait()

        # read result
        val = np.load(OUTPUT_DATA_FILE)
        return val

    def terminate():
        proc.send_signal(signal.CTRL_C_EVENT)

    return f, x0, y0, run_info, terminate


def _wait():
    sig = None
    while sig is None:
        sig = signal.sigtimedwait([signal.SIGQUIT, signal.SIG_UNBLOCK], 1)
        if sig == signal.SIGQUIT:
            raise RuntimeError("Unexpected exception in the objective function process.")
