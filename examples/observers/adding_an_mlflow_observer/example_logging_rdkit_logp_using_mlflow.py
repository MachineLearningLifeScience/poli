"""
This script implements an example of how to use a simple
MLFlow observer (implemented in ./mlflow_observer.py). Running
this script will create a new experiment in ./mlruns.

To run this example, you will need to install mlflow:

    pip install mlflow

To check its results, you will need to start a ui:

    mlflow ui --backend-store-uri ./mlruns
"""

from pathlib import Path

import numpy as np
from mlflow_observer import MlFlowObserver

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # Defining the observer
    TRACKING_URI = THIS_DIR / "mlruns"
    observer = MlFlowObserver(tracking_uri=TRACKING_URI)

    # Initializing a logP objective function.
    problem = objective_factory.create(
        name="rdkit_logp",
        observer=observer,
        string_representation="SELFIES",
        observer_init_info={"run_id": None, "experiment_id": None},
    )
    f, x0 = problem.black_box, problem.x0

    # Logging some examples
    # The observer will register each call to f.
    f(np.array([["[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]", "[C]"]]))

    # Finishing the observer, which will close the MLFlow run.
    observer.finish()
