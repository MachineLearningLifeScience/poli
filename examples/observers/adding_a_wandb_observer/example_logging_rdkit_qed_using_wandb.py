"""
This script uses the wandb observer to log some examples of the rdkit_qed objective function.

To run this example, you will need to install wandb:

    pip install wandb
"""

from pathlib import Path

import numpy as np

from poli.core.problem import Problem
from poli.objective_repository import QEDProblemFactory

from wandb_observer import WandbObserver

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # Defining the observer
    seed = 42
    observer = WandbObserver()

    # Initializing a QED objective function.
    problem = QEDProblemFactory().create(string_representation="SELFIES")
    f, x0 = problem.black_box, problem.x0

    f.set_observer(observer)
    observer.initialize_observer(
        f.info, {"run_id": None, "experiment_id": None}, seed=seed
    )

    y0 = f(x0)

    # Logging some examples
    # The observer will register each call to f.
    f(np.array([["[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]", "[C]"]]))

    # Finishing the observer, which will log a table that's
    # being maintained.
    observer.finish()
