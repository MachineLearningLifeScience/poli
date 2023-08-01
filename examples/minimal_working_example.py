__author__ = "Simon Bartels"

import os

from poli import objective_factory

os.chdir(
    os.path.dirname(__file__)
)  # switch working directory to this folder, just in case
# calling one of the built-in objectives:
problem_info, f, x0, y0, run_info, terminate = objective_factory.create(
    "WHITE_NOISE", caller_info=None
)
print(f(x0[:1, :]))
terminate()
