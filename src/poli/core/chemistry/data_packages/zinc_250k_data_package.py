"""
This script contains a data package that is frequently used in
small molecule optimization: Zinc250k.
"""

from typing import Literal

import numpy as np

from poli.core.data_package import DataPackage


def load_unsupervised_data_for_zinc250k(
    string_representation: Literal["SMILES", "SELFIES"]
) -> np.ndarray:
    # TODO: implement a function that downloads and caches zinc250k.
    ...


class Zinc250kDataPackage(DataPackage):
    def __init__(self, string_representation: Literal["SMILES", "SELFIES"]):
        super().__init__(
            unsupervised_data=load_unsupervised_data_for_zinc250k(
                string_representation=string_representation
            ),
            supervised_data=None,
        )
