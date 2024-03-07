"""Abstract implementation for isolated logic.

This module contains the implementation of an abstract isolated function.
Isolated functions allow you to implement complex, heavily-dependent
objective functions inside isolated processes, which can be run in
their own conda environments. This is useful for running objective
functions that have dependencies that are not compatible with the
dependencies of the main process (e.g. your optimizers, or observers).

AbstractIsolatedFunctions are only expected to implement the __call__
method. This method will (usually) be used to evaluate the function
inside the `_black_box(x, context)` method of black box.

Diving deeper, these isolated functions are used in the `ExternalFunction`
class, which maintains a communication with an isolated process running
the `external_isolated_function_script.py` script.
"""

import numpy as np


class AbstractIsolatedFunction:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        raise NotImplementedError

    def terminate(self):
        pass
