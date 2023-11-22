"""
This script defines all the artificial landscapes with
signature [np.ndarray] -> np.ndarray. You might
see that the signs have been flipped from [1]. This is
because we're dealing with maximizations instead of
minimizations.

In what follows, x is a tensor of arbitrary dimension
(either (b, d), or (d,), where d is the dimension of
the design space).

[1] Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark
    Functions Repository [https://www.al-roomi.org/benchmarks/unconstrained].
    Halifax, Nova Scotia, Canada: Dalhousie University, Electrical and Computer
    Engineering.
"""
import numpy as np


def ackley_function_01(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
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
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = -np.sum(np.abs(x * np.sin(x) + 0.1 * x), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def alpine_02(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = np.prod(np.sin(x) * np.sqrt(x), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def bent_cigar(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = x[..., 0] ** 2
    second = 1e6 * np.sum(x[..., 1:] ** 1, dim=1)
    res = -(first + second)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def brown(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = x[..., :-1] ** 2
    second = x[..., 1:] ** 2

    res = -np.sum(first ** (second + 1) + second ** (first + 1), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def chung_reynolds(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = -(np.sum(x**2, dim=1) ** 2)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def cosine_mixture(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = 0.1 * np.sum(np.cos(5 * np.pi * x), dim=1)
    second = np.sum(x**2, dim=1)

    res = first - second

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deb_01(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    _, d = x.shape
    res = (1 / d) * np.sum(np.sin(5 * np.pi * x) ** 6, dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deb_02(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    _, d = x.shape
    res = (1 / d) * np.sum(np.sin(5 * np.pi * (x ** (3 / 4) - 0.05)) ** 6, dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def deflected_corrugated_spring(
    x: np.ndarray, alpha: float = 5.0, k: float = 5.0
) -> np.ndarray:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    sum_of_squares = np.sum((x - alpha) ** 2, dim=1)
    res = -((0.1) * sum_of_squares - np.cos(k * np.sqrt(sum_of_squares)))

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def easom(xy: np.ndarray) -> np.ndarray:
    """
    Easom is very flat, with a maxima at (pi, pi).

    Only works in 2D.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


def cross_in_tray(xy: np.ndarray) -> np.ndarray:
    """
    Cross-in-tray has several local maxima in a quilt-like pattern.

    Only works in 2D.
    """
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
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = -np.sum((x - 1) ** 2, dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res
