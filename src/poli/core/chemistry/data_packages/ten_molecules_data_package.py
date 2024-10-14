"""
This script contains a data package that is frequently used in
small molecule optimization: Zinc250k.
"""

from typing import Literal

import numpy as np

from poli.core.data_package import DataPackage


class TenMoleculesDataPackage(DataPackage):
    def __init__(self, string_representation: Literal["SMILES", "SELFIES"]):
        if string_representation.upper() == "SMILES":
            ten_molecules = [
                "C",
                "CC",
                "CCC",
                "CCCC",
                "CCCCC",
                "CCCCCC",
                "CCCCCCC",
                "CCCCCCCC",
                "CCCCCCCCC",
                "CCCCCCCCCC",
            ]
        elif string_representation.upper() == "SELFIES":
            ten_molecules = [
                "[C]" * 1,
                "[C]" * 2,
                "[C]" * 3,
                "[C]" * 4,
                "[C]" * 5,
                "[C]" * 6,
                "[C]" * 7,
                "[C]" * 8,
                "[C]" * 9,
                "[C]" * 10,
            ]
        else:
            raise ValueError()

        super().__init__(
            unsupervised_data=np.array(ten_molecules),
            supervised_data=None,
        )
