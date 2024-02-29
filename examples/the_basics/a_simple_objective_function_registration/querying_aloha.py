"""
Once registered, we can create instances of the black box
function. This function is evaluated in an isolated process,
using the conda enviroment we specified at registration.
"""

import numpy as np

from poli import objective_factory

if __name__ == "__main__":
    # Creating an instance of the problem
    f, x0, y0 = objective_factory.create_problem(
        name="our_aloha", observer_init_info=None, observer=None
    )
    print(x0, y0)

    # At this point, you can call f. This will create
    # a new isolated process, where the AlohaBlackBox
    # will run inside the conda environment poli_aloha.
    x1 = np.array(["F", "L", "E", "A", "S"]).reshape(1, -1)
    y1 = f(x1)
    print(x1, y1)
    f.terminate()

    # Another example (using the start function)
    with objective_factory.start(name="our_aloha") as f:
        x = np.array(["F", "L", "E", "A", "S"]).reshape(1, -1)
        y = f(x)
        print(x, y)
