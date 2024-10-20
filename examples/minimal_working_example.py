"""
This is a minimal working example of how to instantiate
a black box objective function. If 'white_noise' is not
registered, then it will be installed from the repository
after asking the user to confirm.
"""

import numpy as np

from poli import objective_factory

problem = objective_factory.create(name="white_noise")
f = problem.black_box

x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].
for _ in range(5):
    print(f"f(x) = {f(x)}")
