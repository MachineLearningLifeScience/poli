"""
A series of testbed functions for optimization. FYI, we changed
the functions from testing maximization instead of minimization.

When run, this script plots the three example functions provided.

See for more examples:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
from typing import Callable, Literal

import numpy as np

from .definitions import (
    easom,
    cross_in_tray,
    shifted_sphere,
    egg_holder,
    ackley_function_01,
    alpine_01,
    alpine_02,
    bent_cigar,
    brown,
    chung_reynolds,
    cosine_mixture,
    deb_01,
    deb_02,
    deflected_corrugated_spring,
)

POSSIBLE_FUNCTIONS = [
    "ackley_function_01",
    "alpine_01",
    "alpine_02",
    "bent_cigar",
    "brown",
    "chung_reynolds",
    "cosine_mixture",
    "deb_01",
    "deb_02",
    "deflected_corrugated_spring",
    "shifted_sphere",
    "easom",
    "cross_in_tray",
    "egg_holder",
]


class ToyContinuousProblem:
    """
    This class will contain the toy objective functions, their limits,
    and the optima location.

    For more information, check definitions.py and [1].

    [1]:  https://al-roomi.org/benchmarks/unconstrained/n-dimensions
    """

    def __init__(
        self,
        name: Literal[
            "ackley_function_01",
            "alpine_01",
            "alpine_02",
            "bent_cigar",
            "brown",
            "chung_reynolds",
            "cosine_mixture",
            "deb_01",
            "deb_02",
            "deflected_corrugated_spring",
            "shifted_sphere",
            "easom",
            "cross_in_tray",
            "egg_holder",
        ],
        n_dims: int = 2,
    ) -> None:
        self.maximize = True
        self.known_optima = True

        if name == "ackley_function_01":
            self.function = ackley_function_01
            self.limits = [-32.0, 32.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "alpine_01":
            self.function = alpine_01
            self.limits = [-10.0, 10.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "alpine_02":
            self.function = alpine_02
            self.limits = [0.0, 10.0]
            self.optima_location = np.array([7.9170526982459462172] * n_dims)
            self.solution_length = n_dims
        elif name == "bent_cigar":
            self.function = bent_cigar
            self.limits = [-100.0, 100.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "brown":
            self.function = brown
            self.limits = [-1.0, 4.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "chung_reynolds":
            self.function = chung_reynolds
            self.limits = [-100.0, 100.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "cosine_mixture":
            self.function = cosine_mixture
            self.limits = [-1.0, 1.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "deb_01":
            self.function = deb_01
            self.limits = [-1.0, 1.0]
            self.optima_location = np.array([0.1] * n_dims)
            self.solution_length = n_dims
        elif name == "deb_02":
            self.function = deb_02
            self.limits = [0.0, 1.0]
            self.optima_location = np.array([0.105 ** (4 / 3)] * n_dims)
            self.solution_length = n_dims
        elif name == "deflected_corrugated_spring":
            self.function = deflected_corrugated_spring
            self.limits = [0.0, 10.0]
            self.optima_location = np.array([5.0] * n_dims)
            self.solution_length = n_dims
        elif name == "shifted_sphere":
            self.function = shifted_sphere
            self.limits = [-4.0, 4.0]
            self.optima_location = np.array([1.0, 1.0])
            self.solution_length = 2
        elif name == "easom":
            self.function = easom
            self.limits = [np.pi - 4, np.pi + 4]
            self.optima_location = np.array([np.pi, np.pi])
            self.solution_length = 2
        elif name == "cross_in_tray":
            self.function = cross_in_tray
            self.limits = [-10, 10]
            self.optima_location = np.array([1.34941, 1.34941])
            self.solution_length = 2
        elif name == "egg_holder":
            self.function = egg_holder
            self.limits = [-700, 700]
            self.optima_location = np.array([512, 404.2319])
            self.solution_length = 2
        else:
            raise ValueError(
                f'Expected {name} to be one of {POSSIBLE_FUNCTIONS}'
            )

        self.optima = self.function(self.optima_location)

    def evaluate_objective(self, x: np.array, **kwargs) -> np.array:
        return self.function(x)

    def __call__(self, x: np.array) -> np.array:
        return self.function(x)
