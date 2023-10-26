"""
In this example, we use RaSP to predict a saturation
mutagenesis for two proteins at the first position.

This example relies on pandas, so remember to
install it in your current environment:

pip install pandas
"""
from pathlib import Path

import numpy as np
import pandas as pd

from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


def saturation_mutagenesis_for_protein_at_position(
    x: np.ndarray, position: int
) -> np.ndarray:
    """
    This function takes a string representation of a
    protein sequence and returns a numpy array of shape
    (20, len(x)) containing the effect of all possible
    single point mutations.
    """
    assert x[position] != ""

    mutations = []
    for amino_acid in AMINO_ACIDS:
        mutation = x.copy()
        mutation[position] = amino_acid
        mutations.append(mutation)

    return np.array(mutations)


if __name__ == "__main__":
    wildtype_pdb_paths_for_rasp = list((THIS_DIR / "two_proteins").glob("*.pdb"))

    _, f_rasp, x0, y0, _ = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=wildtype_pdb_paths_for_rasp,
    )

    # At this point, x0 contains the string
    # representations of the wildtype sequences.
    # Let's construct the saturation mutagenesis
    # of each of these:
    mutations_for_first_protein = saturation_mutagenesis_for_protein_at_position(
        x0[0], 0
    )
    mutations_for_second_protein = saturation_mutagenesis_for_protein_at_position(
        x0[1], 0
    )

    # Now, we can predict the effect of each of these
    # mutations using RaSP:
    x = np.vstack([mutations_for_first_protein, mutations_for_second_protein])

    y = f_rasp(x)

    # Saving the results in a CSV file:
    df = pd.DataFrame(
        [
            {
                "mutation": "".join(x_i),
                "score": y_i,
            }
            for x_i, y_i in zip(x, y.flatten())
        ]
    )

    print(df.head())
