"""
This script uses the wandb observer to log some examples of the rdkit_qed objective function.

To run this example, you will need to install wandb:

    pip install wandb
"""

from pathlib import Path

import numpy as np

from poli import objective_factory

from wandb_observer import WandbObserver

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # Defining the observer
    observer = WandbObserver()

    # Initializing a QED objective function.
    alphabet = ["", "[C]", "..."]
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="rdkit_qed",
        observer=observer,
        alphabet=alphabet,
        string_representation="SELFIES",
        observer_init_info={"run_id": None, "experiment_id": None},
    )

    # Logging some examples
    # The observer will register each call to f.
    f(np.array([["[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]"]]))
    f(np.array([["[C]", "[C]", "[C]", "[C]"]]))

    # Finishing the observer, which will log a table that's
    # being maintained.
    observer.finish()
