"""
This is the main file relevant for users who want to run objective functions.
"""
import os
import signal
import subprocess
import numpy as np


def create(name: str, caller_info):
    cwd = os.path.dirname(__file__)  # execute script in project main folder
    # write caller_info for logger in initialization
    proc = subprocess.Popen("./objective_run_scripts/" + name + ".sh", stdout=None, stderr=None, cwd=cwd)
    signal.sigwait()
    x0, y0 = np.load(init_data_file)
    # TODO: delete file?
    run_info = None

    def f(x: np.ndarray) -> np.ndarray:
        # write x to destination
        np.save(file_name, x)
        # signal process
        proc.send_signal()
        # wait
        signal.sigwait()
        # read result
        val = np.load(result_file_name)
        return val
    return x0, y0, f, run_info
