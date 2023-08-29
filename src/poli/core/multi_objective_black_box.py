"""
Implements a multi_objective version of black box functions,
by this we mean simply concatenating the results of evaluating individual
objective functions.
"""
from typing import List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox


class MultiObjectiveBlackBox(AbstractBlackBox):
    """
    This class implements a multi-objective black box function
    by concatenating the results of evaluating individual
    objective functions.

    TODO: Should we evaluate in parallel?
    """

    def __init__(self, L: int, objective_functions: List[AbstractBlackBox]):
        super().__init__(L)

        self.objective_functions = objective_functions

    def _black_box(self, x, context=None):
        # TODO: Decide on whether we want to evaluate on parallel or not.
        res = []
        for obj_function in self.objective_functions:
            res.append(obj_function(x, context))

        return np.concatenate(res, axis=1)

    def __call__(self, x, context=None):
        """
        This implementation overwrites the call method
        of the parent class to avoid the assertion that
        the output of the black box function is of shape
        [1, 1]. Instead, we allow the output to be [1, n]
        where n is the number of objectives.
        """
        assert len(x.shape) == 2

        # TODO: what happens with batch evaluations that
        # could be processed in parallel?
        f = np.zeros([x.shape[0], len(self.objective_functions)])
        for i, x_i in enumerate(x):
            f_i = self._black_box(x_i.reshape(1, -1), context)  # an [1, n] array
            f[i, :] = f_i
            assert len(f_i.shape) == 2, f"len(f_i.shape)={len(f_i.shape)}, expected 2"
            assert f_i.shape[0] == 1, f"f_i.shape[0]={f_i.shape[0]}, expected 1"
            assert f_i.shape[1] == len(
                self.objective_functions
            ), f"f_i.shape[1]={f_i.shape[1]}, expected {len(self.objective_functions)} (the number of objectives)"
            assert isinstance(f, np.ndarray), f"type(f)={type(f)}, not np.ndarray"

        if self.observer is not None:
            self.observer.observe(x, f, context)

        return f
