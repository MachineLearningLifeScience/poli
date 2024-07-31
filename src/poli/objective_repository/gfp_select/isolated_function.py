from pathlib import Path

import numpy as np
import pandas as pd

from poli.core.abstract_isolated_function import AbstractIsolatedFunction


class GFPSelectIsolatedLogic(AbstractIsolatedFunction):
    def __init__(
        self,
        seed: int = None,
    ):
        gfp_df_path = Path(__file__).parent.resolve() / "assets" / "gfp_data.csv"
        self.seed = seed
        self.gfp_lookup_df = pd.read_csv(gfp_df_path)[
            ["medianBrightness", "aaSequence"]
        ]

        randomized_df = self.gfp_lookup_df.sample(
            frac=1, random_state=seed
        ).reset_index()
        # create 2D array for blackbox evaluation
        x0 = np.array([list(_s) for _s in randomized_df.aaSequence.to_numpy()])

        self.x0 = x0

    def __call__(self, x: np.array, context=None) -> np.ndarray:
        """
        x is string sequence which we look-up in avilable df, return median Brightness
        """
        if isinstance(x, np.ndarray):
            _arr = x.copy()
            x = ["".join(_seq) for _seq in _arr]
        ys = []
        for _x in x:
            seq_subsets = self.gfp_lookup_df[
                self.gfp_lookup_df.aaSequence.str.lower() == _x.lower()
            ]
            # multiple matches possible, shuffle and return one:
            candidate = seq_subsets.sample(n=1, random_state=self.seed)
            ys.append(candidate.medianBrightness)
        return np.array(ys)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        GFPSelectIsolatedLogic,
        name="gfp_select__isolated",
        conda_environment_name="poli__protein_cbas",
    )
