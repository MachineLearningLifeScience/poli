"""Defines several continuous toy problems.

This script defines all the artificial landscapes with
signature [np.ndarray] -> np.ndarray. You might
see that the signs have been flipped from [1] or [2]. This is
because we're dealing with maximizations instead of
minimizations.

[1] Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark
    Functions Repository [https://www.al-roomi.org/benchmarks/unconstrained].
    Halifax, Nova Scotia, Canada: Dalhousie University, Electrical and Computer
    Engineering.
[2] Surjanovic, S. and Bingham, D. Virtual Library of Simulation Experiments:
    Test Functions and Datasets. [https://www.sfu.ca/~ssurjano/optimization.html]
"""

import numpy as np


def ackley_function_01(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    _, d = x.shape

    first = np.exp(-0.2 * np.sqrt((1 / d) * np.sum(x**2, axis=1)))
    second = np.exp((1 / d) * np.sum(np.cos(2 * np.pi * x), axis=1))
    res = 20 * first + second - np.exp(1.0) - 20

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def alpine_01(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    res = -np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def alpine_02(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    res = np.prod(np.sin(x) * np.sqrt(x), axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def bent_cigar(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    first = x[..., 0] ** 2
    second = 1e6 * np.sum(x[..., 1:] ** 1, axis=1)
    res = -(first + second)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def brown(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    first = x[..., :-1] ** 2
    second = x[..., 1:] ** 2

    res = -np.sum(first ** (second + 1) + second ** (first + 1), axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def chung_reynolds(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    res = -(np.sum(x**2, axis=1) ** 2)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def cosine_mixture(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    first = 0.1 * np.sum(np.cos(5 * np.pi * x), axis=1)
    second = np.sum(x**2, axis=1)

    res = first - second

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deb_01(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    _, d = x.shape
    res = (1 / d) * np.sum(np.sin(5 * np.pi * x) ** 6, axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deb_02(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    _, d = x.shape
    res = (1 / d) * np.sum(np.sin(5 * np.pi * (x ** (3 / 4) - 0.05)) ** 6, axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deflected_corrugated_spring(
    x: np.ndarray, alpha: float = 5.0, k: float = 5.0
) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    sum_of_squares = np.sum((x - alpha) ** 2, axis=1)
    res = -((0.1) * sum_of_squares - np.cos(k * np.sqrt(sum_of_squares)))

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def styblinski_tang(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    This function is maximized at (-2.903534, ..., -2.903534), with
    a value of -39.16599 * d.

    If normalize is True, then the function is normalized to be
    maximized at -39.16599 (i.e. we divide the objective by d).
    """
    assert len(x.shape) == 2
    d = x.shape[1]

    y = (x**4) - (16 * (x**2)) + (5 * x)
    if normalize:
        return -0.5 * np.sum(y, axis=1) / d
    else:
        return -0.5 * np.sum(y, axis=1)


def easom(xy: np.ndarray) -> np.ndarray:
    """
    Easom is very flat, with a maxima at (pi, pi).

    Only works in 2D.
    """
    assert len(xy.shape) == 2, "Easom only works in 2D. "
    assert xy.shape[1] == 2, "Easom only works in 2D. "
    x = xy[..., 0]
    y = xy[..., 1]
    return np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


def cross_in_tray(xy: np.ndarray) -> np.ndarray:
    """
    Cross-in-tray has several local maxima in a quilt-like pattern.

    Only works in 2D.
    """
    assert len(xy.shape) == 2, "Easom only works in 2D. "
    assert xy.shape[1] == 2, "Easom only works in 2D. "
    x = xy[..., 0]
    y = xy[..., 1]
    quotient = np.sqrt(x**2 + y**2) / np.pi
    return (
        1 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(10 - quotient))) + 1) ** 0.1
    )


def egg_holder(xy: np.ndarray) -> np.ndarray:
    """
    The egg holder is especially difficult.

    We only know the optima's location in 2D.
    """
    assert len(xy.shape) == 2, "Easom only works in 2D. "
    assert xy.shape[1] == 2, "Easom only works in 2D. "
    x = xy[..., 0]
    y = xy[..., 1]
    return (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) + (
        x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    )


def shifted_sphere(x: np.ndarray) -> np.ndarray:
    """
    The usual squared norm, but shifted away from the origin by a bit.
    Maximized at (1, 1, ..., 1)
    """
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.reshape(-1, x.shape[0])
        batched = False
    else:
        batched = True

    res = -np.sum((x - 1) ** 2, axis=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def camelback_2d(x: np.ndarray) -> np.ndarray:
    """
    Taken directly from the LineBO repository [1].

    [1] https://github.com/kirschnj/LineBO/blob/master/febo/environment/benchmarks/functions.py
    """
    assert len(x.shape) == 2, "Camelback2D only works in 2D. "
    assert x.shape[1] == 2, "Camelback2D only works in 2D. "
    xx = x[:, 0]
    yy = x[:, 1]
    y = (
        (4.0 - 2.1 * xx**2 + (xx**4) / 3.0) * (xx**2)
        + xx * yy
        + (-4.0 + 4 * (yy**2)) * (yy**2)
    )
    return np.maximum(-y, -2.5)


def hartmann_6d(x: np.ndarray) -> np.ndarray:
    """
    The 6 dimensional Hartmann function.

    Since we are aiming to maximize, we negate the output of the
    function.

    Taken from [2].

    [2] Surjanovic, S. and Bingham, D. Virtual Library of Simulation Experiments:
        Test Functions and Datasets. [https://www.sfu.ca/~ssurjano/optimization.html]
    """
    assert len(x.shape) == 2, "Hartmann6D only works in 6D. "
    assert x.shape[1] == 6, "Hartmann6D only works in 6D. "
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    res = []
    # Non-vectorized for now.
    for b in range(x.shape[0]):
        res_at_b = 0
        for i in range(4):
            inner_sum = 0
            for j in range(6):
                inner_sum += A[i][j] * (x[b][j] - P[i][j]) ** 2
            res_at_b += alpha[i] * np.exp(-inner_sum)
        res.append(res_at_b)

    return np.array(res).reshape(-1, 1)


def branin_2d(x: np.ndarray) -> np.ndarray:
    """
    The 2D Branin function.

    Taken from [2]. Notice that we negate the output of the function
    since we are aiming to maximize instead of minimizing.

    [2] Surjanovic, S. and Bingham, D. Virtual Library of Simulation Experiments:
        Test Functions and Datasets. [https://www.sfu.ca/~ssurjano/optimization.html]
    """
    assert len(x.shape) == 2, "Branin2D only works in 2D. "
    assert x.shape[1] == 2, "Branin2D only works in 2D. "
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return -y


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0):
    """
    Compute the Rosenbrock function.

    Parameters
    ----------
    x (np.ndarray): A 2D numpy array of shape [b, d] where 'b' is the batch size and 'd' is the dimensionality.
    a (float): The 'a' parameter for the Rosenbrock function. Default is 1.
    b (float): The 'b' parameter for the Rosenbrock function. Default is 100.

    Returns
    -------
    np.ndarray: The value of the Rosenbrock function at each point in 'x'. Shape is [b,].
    """
    assert len(x.shape) == 2, "Input x must be a 2D array."
    d = x.shape[1]
    assert d > 1, "Dimensionality must be greater than 1."

    return -(
        np.sum((a - x[:, :-1]) ** 2 + b * (x[:, 1:] - x[:, :-1] ** 2) ** 2, axis=1)
    )


def levy(x: np.ndarray):
    """
    Compute the Levy function.

    Parameters
    ----------
    x (np.ndarray): A 2D numpy array of shape [b, d] where 'b' is the batch size and 'd' is the dimensionality.

    Returns
    -------
    np.ndarray: The value of the Levy function at each point in 'x'. Shape is [b,].

    References
    ----------
    [1] Surjanovic, S. and Bingham, D. Virtual Library of Simulation Experiments:
    Test Functions and Datasets. [https://www.sfu.ca/~ssurjano/optimization.html]
    """
    assert len(x.shape) == 2, "Input x must be a 2D array."
    d = x.shape[1]
    assert d > 0, "Dimensionality must be greater than 0."

    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[:, 0])) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[:, -1])) ** 2)

    wi = w[:, :-1]
    term2 = np.sum((wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2), axis=1)

    return -(term1 + term2 + term3)


if __name__ == "__main__":
    b = branin_2d
    maximal_b = b(np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]))
