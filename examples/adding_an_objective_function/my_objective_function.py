__author__ = 'Simon Bartels'

import os
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
        AA = ['a', 'r', 'n', 'd']
        AA_IDX = {AA[i]: i for i in range(len(AA))}
        return ProblemSetupInformation(name="MY_PROBLEM", max_sequence_length=3, aligned=True, alphabet=AA_IDX)

    def create(self) -> (AbstractBlackBox, np.ndarray, np.ndarray):
        L = self.get_setup_information().get_max_sequence_length()
        f = MyBlackBox(L)
        x0 = np.ones([1, L])
        return f, x0, f(x0)


"""
Step 3: Write a small shell script such as `my_objective_function.sh` which calls this file.
(The encapsulation in a shell script allows for example to load a different python environment.)

Done.
"""
if __name__ == '__main__':
    from poli.objective import run
    problem_factory_name = os.path.basename(__file__)[:-2] + MyProblemFactory.__name__
    run(problem_factory_name)
