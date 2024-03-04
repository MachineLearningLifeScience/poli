"""
This script shows how to use the toy continuous
black box objectives inside poli.
"""

import numpy as np

from poli import objective_factory

if __name__ == "__main__":
    ackley_function, _, _ = objective_factory.create(
        "toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=2,
    )

    print(ackley_function(np.array([[0.0, 0.0]])))
