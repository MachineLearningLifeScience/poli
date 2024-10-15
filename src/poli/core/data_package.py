"""
Implements a data package, containing unsupervised and
supervised data.
"""

from __future__ import annotations

import numpy as np


class DataPackage:
    def __init__(
        self,
        unsupervised_data: np.ndarray | None,
        supervised_data: tuple[np.ndarray, np.ndarray] | None,
    ):
        self.unsupervised_data = unsupervised_data
        self.supervised_data = supervised_data
