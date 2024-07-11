"""This script instantiates the observer manually.

The main purpose of poli is to allow seamless combination of different problems, algorithms and observers.
This is why we encourage users to leave the instantiation of the observer to poli.
However, the following script is maybe more intuitive and it shows what is happening in the background when all components can run in the same environment.
"""

import numpy as np
from print_observer import SimplePrintObserver

from poli import objective_factory
from poli.core.registry import DEFAULT_OBSERVER_NAME

if __name__ == "__main__":
    # Instantiate the objective
    problem = objective_factory.create(
        name="aloha",
        observer_name=DEFAULT_OBSERVER_NAME,  # instantiates the default observer which does nothing
    )

    observer = SimplePrintObserver()
    observer_info = observer.initialize_observer(
        seed=0, problem_setup_info=problem.black_box_information, caller_info=dict()
    )

    f = problem.black_box
    f.set_observer(observer)

    # Run the objective. Each objective call
    # is registered by the observer.
    f(np.array([list("MIGUE")]))
    f(np.array([list("FLEAS")]))
    f(np.array([list("ALOHA")]))

    # An algorithm may also send information to the observer via the AlgorithmObserverWrapper
    problem.observer.log({"ALGORITHM": "no_algo", "OTHER_INFO": "nothing"})
