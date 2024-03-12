"""
A series of testbed functions for optimization.

See for more examples:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from typing import Literal

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
    camelback_2d,
    styblinski_tang,
)

# Notice: these will be used by pytest to test the
# instancing of the toy objective functions.
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
    "styblinski_tang",
    "shifted_sphere",
    "easom",
    "cross_in_tray",
    "egg_holder",
    "camelback_2d",
]
TWO_DIMENSIONAL_PROBLEMS = [
    "shifted_sphere",
    "easom",
    "cross_in_tray",
    "egg_holder",
    "camelback_2d",
]


class ToyContinuousProblem:
    """
    Contains the toy objective functions, their limits, and the optima location.

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
            "styblinski_tang",
            "shifted_sphere",
            "easom",
            "cross_in_tray",
            "egg_holder",
            "camelback_2d",
        ],
        n_dims: int = 2,
        embed_in: int = None,
    ) -> None:
        self.maximize = True
        self.known_optima = True
        self.dimensions_to_embed_in = None

        if n_dims != 2 and name in TWO_DIMENSIONAL_PROBLEMS:
            if embed_in is None:
                raise ValueError(
                    f"Function {name} can only be instantiated in two dimensions (received {n_dims})."
                    " Alternatively, you can embed the function in higher dimensions by setting"
                    " embed_in: int to the desired dimension. When doing so, the 2 dimensions will be "
                    "randomly selected among the embed_in."
                )

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
        elif name == "styblinski_tang":
            self.function = styblinski_tang
            self.limits = [-5.0, 5.0]
            self.optima_location = np.array([-2.903534] * n_dims)
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
        elif name == "camelback_2d":
            self.function = camelback_2d
            self.limits = [-5, 5]
            self.optima_location = np.array([0.0898, -0.7126])
            self.solution_length = 2
        else:
            raise ValueError(f"Expected {name} to be one of {POSSIBLE_FUNCTIONS}")

        self.optima = self.function(self.optima_location.reshape(1, -1))

        # If embed_in is not None, then we will embed the
        # function in embed_in dimensions. This is useful for testing
        # algorithms that leverage low intrinsic dimensionality.
        if embed_in is not None:
            assert n_dims < embed_in, (
                f"Expected the intrinsic dimensionality of the problem to be lower than the "
                f"dimensionality of the space, but got {self.solution_length} and {n_dims} respectively."
            )

            self.dimensions_to_embed_in = np.random.permutation(embed_in)[:n_dims]
            self.solution_length = embed_in
            self.limits = [self.limits[0]] * embed_in, [self.limits[1]] * embed_in
            previous_optima_location = self.optima_location.copy()
            self.optima_location = np.zeros(embed_in)
            self.optima_location[self.dimensions_to_embed_in] = previous_optima_location

            _current_function = self.function
            self.function = lambda x: _current_function(
                x[:, self.dimensions_to_embed_in]
            )
        else:
            # We need to make sure that the user specified a feasible
            # number of dimensions. If not, we raise an error.
            assert n_dims == self.solution_length, (
                f"The solution length for {name} should be {self.solution_length},"
                f" but received {n_dims}."
            )

    def evaluate_objective(self, x: np.array, **kwargs) -> np.array:
        return self.function(x)

    def __call__(self, x: np.array) -> np.array:
        return self.function(x).reshape(-1, 1)
