"""
This script contains a data package that is frequently used in
small molecule optimization: Zinc250k.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from poli.core.data_package import DataPackage
from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies


class RandomMoleculesDataPackage(DataPackage):
    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"],
        n_molecules: int = 10,
        seed: int | None = None,
    ):
        assert (
            n_molecules <= 5000
        ), "This data package has been implemented for up to 5000 random molecules."
        CHEMISTRY_DATA_PACKAGES_DIR = Path(__file__).parent
        five_thousand_molecules = np.load(
            CHEMISTRY_DATA_PACKAGES_DIR / "five_thousand_smiles.npz",
            allow_pickle=True,
        )["x"]

        if string_representation.upper() == "SELFIES":
            five_thousand_molecules_ = translate_smiles_to_selfies(
                five_thousand_molecules,
                strict=True,
            )
            five_thousand_molecules = np.array(five_thousand_molecules_)

        if seed is not None:
            np.random.seed(seed)

        unsupervised_data = np.random.choice(
            five_thousand_molecules, (n_molecules,), replace=False
        )
        supervised_data = None

        super().__init__(unsupervised_data, supervised_data)
