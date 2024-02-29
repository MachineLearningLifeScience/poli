"""
Implements an abstract problem.

Problems are standardized ways of running benchmarks. They are used to
create black boxes and to run optimization algorithms.

At its core, a problem is a black box with a known setup. This setup
includes the following information:

- The initial evaluation(s) x0 and y0.
- The fidelity of the black box (e.g., "high" or "low").
- The black box function and its information (e.g. whether it is
  noisy, continuous or discrete, etc.).
- An evaluation budget, which is the maximum number of evaluations
  allowed.
"""

from typing import Literal

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation


class Problem:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        fidelity: Literal["high", "low"] = "high",
        evaluation_budget: int = float("inf"),
    ):
        self.black_box = black_box
        self.x0 = x0
        self.fidelity = fidelity
        self.evaluation_budget = evaluation_budget
        self.black_box_information = black_box.info
        self._validate()

    def _validate(self):
        if self.evaluation_budget < 0:
            raise ValueError("Evaluation budget must be non-negative.")
        if self.fidelity not in ["high", "low"]:
            raise ValueError("Fidelity must be either 'high' or 'low'.")
        if not isinstance(self.black_box, AbstractBlackBox):
            raise ValueError("Black box must be an instance of AbstractBlackBox.")
        if not isinstance(self.x0, np.ndarray):
            raise ValueError("x0 must be a numpy array.")
        # TODO: validate whether self.x0 is of the right shape.

    def is_discrete(self):
        return self.black_box_information.discrete

    def is_deterministic(self):
        return self.black_box_information.deterministic

    def is_continuous(self):
        return not self.is_discrete()
