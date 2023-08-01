__author__ = "Simon Bartels"

from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation


"""
Adding a custom objective function requires three steps.
Step 1: Write a class that inherits from AbstractBlackBox and implement _black_box
"""


class MyBlackBox(AbstractBlackBox):
    def _black_box(self, x, context=None):
        # the returned value must be a [1, 1] array
        return np.sum(x).reshape(-1, 1)


"""
Step 2: Inherit from AbstractProblemFactory and implement the two abstract methods accordingly.
"""


class MyProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        AA = ["a", "r", "n", "d"]
        AA_IDX = {AA[i]: i for i in range(len(AA))}
        return ProblemSetupInformation(
            name="MY_PROBLEM", max_sequence_length=3, aligned=True, alphabet=AA_IDX
        )

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        L = self.get_setup_information().get_max_sequence_length()
        f = MyBlackBox(L)
        x0 = np.ones([1, L])
        return f, x0, f(x0)


if __name__ == "__main__":
    from poli import objective_factory
    from poli.core.registry import register_problem

    # (once) we have to register our factory
    my_problem_factory = MyProblemFactory()
    register_problem(
        my_problem_factory,
        conda_environment_location="/Users/migd/anaconda3/envs/poli-dev",
    )

    # now we can instantiate our objective
    problem_name = my_problem_factory.get_setup_information().get_problem_name()
    problem_info, f, x0, y0, run_info = objective_factory.create(
        problem_name, caller_info=None, observer=None
    )
    print(f(x0[:1, :]))
    f.terminate()
