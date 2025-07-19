"""
This module implements TrpB DMS,
using the open-available data from [2].

The black-box is a lookup of the combinatorically complete mutation landscape.
If the proposed sequence exists the associated value is returned.

[1] "A combinatorially complete epistatic fitness landscape in an enzyme active site"
Kadina E. Johnston, Patrick J. Almhjell, Ella J. Watkins-Dulaney, Grace Liu, Nicholas J. Porter, Jason Yang, and Frances H. Arnold.
doi: https://doi.org/10.1073/pnas.2400439121 .
July 29, 2024.


"""

# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from __future__ import annotations

from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np
import pandas as pd

from poli.core.abstract_isolated_function import AbstractIsolatedFunction

THIS_DIR = Path(__file__).parent.resolve()


class DMSTrpBIsolatedLogic(AbstractIsolatedFunction):
    """
    TrpB internal implementation.

    Parameters
    ----------
    experiment_id : str, optional
        The experiment ID, by default None.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the stability of the mutant(s) in x.
    _load_dms_data()
        This function loads the DMS data under assets.

    Notes
    -----
    - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
    """

    def __init__(
        self,
        experiment_id: str | None = None,
    ):
        """
        Initialize the GB1 Register object.

        Parameters:
        -----------

        experiment_id : str, optional
            The experiment ID, by default None.

        Notes:
        ------
        - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
        """

        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

        # see [1] for reference:
        self.wt = "VFVS"
        self.positions = [183, 184, 227, 228]  # included for completeness

        self.dms_df = self._load_dms_data()

        self.x0 = np.array(list(self.wt))[None, :]

    def _load_dms_data(self) -> pd.DataFrame:
        return pd.read_csv(THIS_DIR / "assets" / "fitness.csv")

    def _return_dms_val(self, x) -> float:
        return self.dms_df[self.dms_df.Combo == x].fitness.values[0]

    def __call__(self, x, context=None):
        """
        Computes the fitness of the variant(s) in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            The fitness of the mutant(s) in x.

        Notes
        -----
        - x is a np.array[str] of shape [b, L], where L is the length
          of the longest sequence in the batch, and b is the batch size.

        Throws
        -----
        - ValueError exception if input is invalid.
        """
        results = []
        for i, x_i in enumerate(x):
            # Assuming x_i is an array of strings
            if len(x_i) != 4:
                raise ValueError(
                    f"Inputs of L=4 expected!\n Curren input length={len(x_i)} at index {i}"
                )
            mutant_residue_string = "".join(x_i)
            result = self._return_dms_val(mutant_residue_string)
            results.append(result)

        return np.array(results).reshape(-1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        DMSTrpBIsolatedLogic,
        name="dms_trpb__isolated",
        conda_environment_name="poli__dms",
        force=True,
    )
