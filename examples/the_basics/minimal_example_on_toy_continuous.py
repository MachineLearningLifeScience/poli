"""
This script shows how to use the toy continuous
black box objectives inside poli.
"""

import numpy as np

from poli import objective_factory
from poli.objective_repository import ToyContinuousBlackBox

if __name__ == "__main__":
    # One way
    ackley_problem = objective_factory.create(
        "toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=2,
    )
    ackley_function = ackley_problem.black_box

    # Another way
    ackley_function_2 = ToyContinuousBlackBox(
        function_name="ackley_function_01",
        n_dimensions=2,
    )

    print(ackley_function(np.array([[0.0, 0.0]])))
    print(ackley_function_2(np.array([[0.0, 0.0]])))
