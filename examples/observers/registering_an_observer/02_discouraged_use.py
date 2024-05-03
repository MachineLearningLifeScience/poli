"""This script takes the registered observer and instantiates it.

To showcase isolation, we run this script in a _different_
environment to the one used to register and run the observer,
one that doesn't have e.g. wandb installed.
"""

import numpy as np

from poli import objective_factory

if __name__ == "__main__":
    # Instantiate the objective
    problem = objective_factory.create(
        name="aloha",
        observer_name="simple_print_observer",  # instantiate the registered observer
    )
    f = problem.black_box

    # Run the objective. Each objective call
    # is registered by the observer.
    f(np.array([list("MIGUE")]))
    f(np.array([list("FLEAS")]))
    f(np.array([list("ALOHA")]))

    # An algorithm may also send information to the observer via the AlgorithmObserverWrapper
    problem.observer.log({"ALGORITHM": "no_algo", "OTHER_INFO": "nothing"})
