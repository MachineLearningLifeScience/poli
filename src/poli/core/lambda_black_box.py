"""
This module implements a lambda black box:

Once the user has specified a callable f which
takes as input np.ndarray and returns np.ndarray,
this black box uses it for the evaluation.

This is useful when the user has already implemented
a lot of functionality around a callable, and wants to
leverage all that comes with a poli black box: logging,
budget management, etc.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation


class LambdaBlackBox(AbstractBlackBox):
    """
    A Lambda black box is a black box that uses a callable
    for the evaluation.

    Parameters:
    -----------
    function : Callable[[np.ndarray], np.ndarray]
        The callable that will be used for the evaluation.
    info : BlackBoxInformation, optional
        The information about the black box. If None,
        a default information is used.
    batch_size : int, optional
        The batch size for evaluating the black box function. Default is None.
    parallelize : bool, optional
        Flag indicating whether to evaluate the black box function in parallel.
        Default is False.
    num_workers : int, optional
        The number of workers to use for parallel evaluation. Default is None,
        which uses half of the available CPU cores.
    evaluation_budget : int, optional
        The maximum number of evaluations allowed for the black box function.
        Default is float("inf").
    force_isolation : bool, optional
        Flag indicating whether to force isolation of the black
        box function. In this black box, this flag is ignored.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        info: BlackBoxInformation | None = None,
        batch_size=None,
        parallelize=False,
        num_workers=None,
        evaluation_budget=float("inf"),
        force_isolation=False,
    ):
        super().__init__(
            batch_size, parallelize, num_workers, evaluation_budget, force_isolation
        )
        self.function = function
        self._information = (
            info
            if info is not None
            else BlackBoxInformation(
                name="lambda_function",
                max_sequence_length=np.inf,
                aligned=False,
                fixed_length=False,
                deterministic=False,
                alphabet=None,
                discrete=True,
            )
        )

    def _black_box(self, x, context=None):
        return self.function(x)

    def get_black_box_info(self):
        return self._information
