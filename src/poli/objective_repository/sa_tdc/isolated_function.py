"""
Implements a synthetic-accessibility objective using the TDC oracles [1].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
"""

from poli.core.chemistry.tdc_isolated_function import TDCIsolatedFunction


class SAIsolatedLogic(TDCIsolatedFunction):
    """Synthetic-accessibility black box implementation using the TDC oracles [1].

    Parameters
    ----------
    from_smiles : bool, optional
        Flag indicating whether to use SMILES strings as input, by default True.
    """

    def __init__(
        self,
        from_smiles: bool = True,
    ):
        """
        Initialize the SABlackBox object.

        Parameters
        ----------
        from_smiles : bool, optional
            Flag indicating whether to use SMILES strings as input, by default True.
        """
        oracle_name = "SA"
        super().__init__(
            oracle_name=oracle_name,
            from_smiles=from_smiles,
        )


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        SAIsolatedLogic,
        name="sa_tdc__isolated",
        conda_environment_name="poli__tdc",
        force=True,
    )
