"""
This module implements a lambda black box:

Once the user has specified a callable f which
takes as input np.ndarray and returns np.ndarray,
this black box uses it for the evaluation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation


class LambdaBlackBox(AbstractBlackBox):
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
