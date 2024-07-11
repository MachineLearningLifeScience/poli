"""
A series of testbed functions for optimization.

See for more examples:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from typing import List, Literal

import numpy as np

from .definitions import (
    ackley_function_01,
    alpine_01,
    alpine_02,
    bent_cigar,
    branin_2d,
    brown,
    camelback_2d,
    chung_reynolds,
    cosine_mixture,
    cross_in_tray,
    deb_01,
    deb_02,
    deflected_corrugated_spring,
    easom,
    egg_holder,
    hartmann_6d,
    levy,
    rosenbrock,
    shifted_sphere,
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
    "hartmann_6d",
    "branin_2d",
    "rosenbrock",
    "levy",
]
TWO_DIMENSIONAL_PROBLEMS = [
    "easom",
    "cross_in_tray",
    "egg_holder",
    "camelback_2d",
    "branin_2d",
]
SIX_DIMENSIONAL_PROBLEMS = ["hartmann_6d"]


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
            "branin_2d",
            "hartmann_6d",
            "rosenbrock",
            "levy",
        ],
        n_dims: int = 2,
        embed_in: int = None,
        dimensions_to_embed_in: List[int] = None,
    ) -> None:
        self.maximize = True
        self.known_optima = True
        self.dimensions_to_embed_in = dimensions_to_embed_in

        if n_dims != 2 and name in TWO_DIMENSIONAL_PROBLEMS:
            if embed_in is None:
                raise ValueError(
                    f"Function {name} can only be instantiated in two dimensions (received {n_dims})."
                    " Alternatively, you can embed the function in higher dimensions by setting"
                    " embed_in: int to the desired dimension. When doing so, the 2 dimensions will be "
                    "randomly selected among the embed_in."
                )

        if dimensions_to_embed_in is not None:
            assert (
                embed_in is not None
            ), "Expected dimensions_to_embed_in to be None if embed_in is None."
            for dim in dimensions_to_embed_in:
                if dim >= embed_in or dim < 0:
                    raise ValueError(
                        f"Dimension to embed in {dim} is higher than the number of dimensions {n_dims} or negative."
                    )

        if name == "ackley_function_01":
            self.function = ackley_function_01
            self.limits = [-32.0, 32.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[15.0] * self.solution_length])
        elif name == "alpine_01":
            self.function = alpine_01
            self.limits = [-10.0, 10.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[8.0] * self.solution_length])
        elif name == "alpine_02":
            self.function = alpine_02
            self.limits = [0.0, 10.0]
            self.optima_location = np.array([7.9170526982459462172] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[3.0] * self.solution_length])
        elif name == "bent_cigar":
            self.function = bent_cigar
            self.limits = [-100.0, 100.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[50.0] * self.solution_length])
        elif name == "brown":
            self.function = brown
            self.limits = [-1.0, 4.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[3.0] * self.solution_length])
        elif name == "chung_reynolds":
            self.function = chung_reynolds
            self.limits = [-100.0, 100.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[50.0] * self.solution_length])
        elif name == "cosine_mixture":
            self.function = cosine_mixture
            self.limits = [-1.0, 1.0]
            self.optima_location = np.array([0.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[-0.75] * self.solution_length])
        elif name == "deb_01":
            self.function = deb_01
            self.limits = [-1.0, 1.0]
            self.optima_location = np.array([0.1] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[-0.75] * self.solution_length])
        elif name == "deb_02":
            self.function = deb_02
            self.limits = [0.0, 1.0]
            self.optima_location = np.array([0.105 ** (4 / 3)] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[0.75] * self.solution_length])
        elif name == "deflected_corrugated_spring":
            self.function = deflected_corrugated_spring
            self.limits = [0.0, 10.0]
            self.optima_location = np.array([5.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[1.0] * self.solution_length])
        elif name == "styblinski_tang":
            self.function = styblinski_tang
            self.limits = [-5.0, 5.0]
            self.optima_location = np.array([-2.903534] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[4.0] * self.solution_length])
        elif name == "shifted_sphere":
            self.function = shifted_sphere
            self.limits = [-4.0, 4.0]
            self.optima_location = np.array([1.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[-3.0] * self.solution_length])
        elif name == "easom":
            self.function = easom
            self.limits = [np.pi - 4, np.pi + 4]
            self.optima_location = np.array([np.pi, np.pi])
            self.solution_length = 2
            self.x0 = np.array([[0.0] * self.solution_length])
        elif name == "cross_in_tray":
            self.function = cross_in_tray
            self.limits = [-10.0, 10.0]
            self.optima_location = np.array([1.34941, 1.34941])
            self.solution_length = 2
            self.x0 = np.array([[-7.0] * self.solution_length])
        elif name == "egg_holder":
            self.function = egg_holder
            self.limits = [-700.0, 700.0]
            self.optima_location = np.array([512, 404.2319])
            self.solution_length = 2
            self.x0 = np.array([[-600.0] * self.solution_length])
        elif name == "camelback_2d":
            self.function = camelback_2d
            self.limits = [-5.0, 5.0]
            self.optima_location = np.array([0.0898, -0.7126])
            self.solution_length = 2
            self.x0 = np.array([[-3.0] * self.solution_length])
        elif name == "hartmann_6d":
            self.function = hartmann_6d
            self.limits = [0.0, 1.0]
            self.optima_location = np.array(
                [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
            )
            self.solution_length = 6
            self.x0 = np.array([[0.8] * self.solution_length])
        elif name == "branin_2d":
            self.function = branin_2d
            self.limits = [-5.0, 15.0]
            self.optima_location = np.array([9.42478, 2.475])
            self.solution_length = 2
            self.x0 = np.array([0.0] * n_dims).reshape(1, n_dims)
        elif name == "rosenbrock":
            self.function = rosenbrock
            self.limits = [-5.0, 10.0]
            self.optima_location = np.array([1.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[0.0] * self.solution_length])
        elif name == "levy":
            self.function = levy
            self.limits = [-10.0, 10.0]
            self.optima_location = np.array([1.0] * n_dims)
            self.solution_length = n_dims
            self.x0 = np.array([[0.0] * self.solution_length])
        else:
            raise ValueError(f"Expected {name} to be one of {POSSIBLE_FUNCTIONS}")

        self.optima = self.function(self.optima_location.reshape(1, -1))

        # If embed_in is not None, then we will embed the
        # function in embed_in dimensions. This is useful for testing
        # algorithms that leverage low intrinsic dimensionality.
        # At this point, solution length is the intrinsic dimensionality
        # of the problem.
        if embed_in is not None:
            assert self.solution_length < embed_in, (
                f"Expected the intrinsic dimensionality of the problem to be lower than the "
                f"dimensionality of the space, but got {self.solution_length} and {embed_in} respectively."
            )

            if dimensions_to_embed_in is None:
                self.dimensions_to_embed_in = np.random.permutation(embed_in)[
                    : self.solution_length
                ]

            # We update the solution length to the embedded dimensionality.
            self.solution_length = embed_in
            previous_optima_location = self.optima_location.copy()
            previous_x0 = self.x0.copy()
            self.optima_location = np.zeros(embed_in)
            self.optima_location[self.dimensions_to_embed_in] = previous_optima_location
            self.x0 = np.zeros(embed_in)
            self.x0[self.dimensions_to_embed_in] = previous_x0.flatten()
            self.x0 = self.x0.reshape(1, -1)

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
