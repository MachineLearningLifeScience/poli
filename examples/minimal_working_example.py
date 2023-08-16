"""
This is a minimal working example of how to instantiate
a black box objective function. If 'white_noise' is not
registered, then it will be installed from the repository
after asking the user to confirm.
"""
import numpy as np
from poli import objective_factory

problem_info, f, x0, y0, run_info = objective_factory.create(name="aloha")

x = np.array([[1]])  # must be of shape [b, d], in this case [1, 1].
for _ in range(10):
    print(f"f(x) = {f(x)}")
