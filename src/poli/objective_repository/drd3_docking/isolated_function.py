from poli.core.chemistry.tdc_isolated_function import TDCIsolatedFunction
from poli.core.registry import register_isolated_function


class DRD3IsolatedFunction(TDCIsolatedFunction):
    """
    DRD3BlackBox is a class that represents a black box for DRD3 docking.

    Parameters
    ----------
    from_smiles : bool, optional
        Flag indicating whether to use SMILES strings as input, by default True.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, info, batch_size=None, parallelize=False, num_workers=None, from_smiles=True)
        Initializes a new instance of the DRD3BlackBox class.
    """

    def __init__(
        self,
        from_smiles: bool = True,
    ):
        oracle_name = "3pbl_docking"
        super().__init__(
            oracle_name=oracle_name,
            from_smiles=from_smiles,
        )


if __name__ == "__main__":
    # One example of loading up this black box:
    drd3_isolated_function = DRD3IsolatedFunction(from_smiles=True)

    register_isolated_function(
        drd3_isolated_function,
        name="drd3_docking__isolated",
        conda_environment_name="poli__tdc",
    )
