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

    def __repr__(self) -> str:
        unsupervised_data = (
            self.unsupervised_data.shape if self.unsupervised_data is not None else None
        )
        supervised_data = (
            self.supervised_data[0].shape if self.supervised_data is not None else None
        )
        return f"DataPackage(unsupervised_data={unsupervised_data}, supervised_data={supervised_data})"

    def __str__(self) -> str:
        return self.__repr__()
