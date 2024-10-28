"""
This script contains a data package that is frequently used in
small molecule optimization: sampling random molecules from Zinc250k.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np

from poli.core.data_package import DataPackage
from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies


class RandomMoleculesDataPackage(DataPackage):
    """
    Returns a supervised data package with random molecules from Zinc250k.

    We subsampled 5000 smiles from Zinc250k and stored them in a numpy file,
    and this data package samples n_molecules from this set.

    Parameters
    ----------
    string_representation : Literal["SMILES", "SELFIES"]
        The string representation of the molecules.
    n_molecules : int, optional
        The number of molecules to sample from the dataset, by default 10.
    seed : int, optional
        The seed for the random number generator, by default None.
        If provided, we seed numpy random number generator with this seed.
    tokenize_with : Callable[[str], list[str]], optional
        A function that tokenizes the molecules, by default None.
        If provided, we tokenize the molecules with this function.
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"],
        n_molecules: int = 10,
        seed: int | None = None,
        tokenize_with: Callable[[str], list[str]] = None,
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

        if tokenize_with is not None:
            unsupervised_data = np.array(
                [tokenize_with(mol) for mol in unsupervised_data if mol is not None]
            )

        super().__init__(unsupervised_data, supervised_data)
