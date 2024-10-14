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

from __future__ import annotations

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.data_package import DataPackage
from poli.core.util.algorithm_observer_wrapper import AlgorithmObserverWrapper
from poli.core.util.default_observer import DefaultObserver


class Problem:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        data_package: DataPackage | None = None,
        strict_validation: bool = True,
    ):
        self.black_box: AbstractBlackBox = black_box
        self.x0: np.ndarray = x0
        self.data_package: DataPackage | None = data_package
        self.black_box_information = black_box.info
        if strict_validation:
            self._validate()
        self.observer: AlgorithmObserverWrapper = AlgorithmObserverWrapper(
            DefaultObserver()
        )
        self.observer_info = None

    def _validate(self):
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

    def set_observer(self, observer: AlgorithmObserverWrapper, observer_info: object):
        self.observer = observer
        self.observer_info = observer_info

    @property
    def info(self):
        return self.black_box.info

    def __str__(self):
        return f"<Problem: {self.black_box.info.get_problem_name()} x0: {self.x0} data_package: {self.data_package}>"

    def __repr__(self):
        return f"<Problem: {self.black_box.info.get_problem_name()} x0: {self.x0} data_package: {self.data_package} discrete={self.is_discrete()} deterministic={self.is_deterministic()}>"
