from typing import Literal

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox

from poli.core.util.isolation.instancing import instance_function_as_isolated_process

AVAILABLE_TDC_ORACLES = [
    "DRD2",
    "DRD3",
    "GSK3Beta",
    "JNK3",
]


class TDCBlackBox(AbstractBlackBox):
    """
    An abstract black box for the TDC (Therapeutics Data
    Commons) problems.

    Parameters
    ----------
    oracle_name: str
        The name of the oracle to be used.
    string_representation : Literal["SMILES", "SELFIES"], optional
        A string (either "SMILES" or "SELFIES") specifying which
        molecule representation you plan to use.
    batch_size : int, optional
        The batch size for simultaneous execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    force_isolation: bool, optional
        Whether to force the isolation of the black box. Default is False.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, info, batch_size=None, parallelize=False, num_workers=None, from_smiles=True)
        Initializes a new instance of the DRD3BlackBox class.

    References
    ----------
    [1] Artificial intelligence foundation for therapeutic science.
        Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
    """

    def __init__(
        self,
        oracle_name: Literal["DRD2", "3pbl_docking", "GSK3B"] = "DRD2",
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        force_isolation: bool = False,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        **kwargs_for_oracle,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.oracle_name = oracle_name

        from_smiles = string_representation.upper() == "SMILES"
        if not force_isolation:
            try:
                from poli.core.chemistry.tdc_isolated_function import (
                    TDCIsolatedFunction,
                )

                self.inner_function = TDCIsolatedFunction(
                    oracle_name=oracle_name,
                    from_smiles=from_smiles,
                    **kwargs_for_oracle,
                )
            except ImportError:
                self.inner_function = instance_function_as_isolated_process(
                    name="tdc__isolated",
                    oracle_name=oracle_name,
                    from_smiles=from_smiles,
                    **kwargs_for_oracle,
                )
        else:
            self.inner_function = instance_function_as_isolated_process(
                name="tdc__isolated",
                oracle_name=oracle_name,
                from_smiles=from_smiles,
                **kwargs_for_oracle,
            )

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        return self.inner_function(x, context)
