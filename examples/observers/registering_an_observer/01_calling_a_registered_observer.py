"""This script takes the registered observer and instantiates it.

To showcase isolation, we run this script in a _different_
environment to the one used to register and run the observer,
one that doesn't have e.g. wandb installed.
"""

import numpy as np

from poli import objective_factory
from poli.core.util.external_observer import ExternalObserver

if __name__ == "__main__":
    # Instantiate the objective
    problem = objective_factory.create(
        name="aloha",
        observer_name="wandb",    # instantiate the registered observer
    )
    f = problem.black_box

    # Run the objective. Each objective call
    # is registered by the observer (check
    # ./wandb_observer.py for details).
    f(np.array([list("MIGUE")]))
    f(np.array([list("FLEAS")]))
    f(np.array([list("ALOHA")]))
