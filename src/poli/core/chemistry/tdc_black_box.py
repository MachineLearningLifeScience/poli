"""
Implements an abstract TDC black box for all the PMO [1, 2] related problems.

PMO stands for Practical Molecular Optimization, a benchmark
suite for molecular optimization problems that extends GuacaMol [3].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
[2] Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization
    Wenhao Gao, Tianfan Fu, Jimeng Sun, Connor W. Coley
    https://arxiv.org/abs/2206.12411
[3] GuacaMol: benchmarking models for de novo molecular design.
    Brown, N. et al.  J Chem Inf Model 59 (2019).
    https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.isolation.instancing import get_inner_function


class TDCBlackBox(AbstractBlackBox):
    """
    An abstract black box for the TDC (Therapeutics Data
    Commons) problems [1].

    Parameters
    ----------
    oracle_name: str
        The name of the oracle to be used.
    string_representation : Literal["SMILES", "SELFIES"], optional
        A string (either "SMILES" or "SELFIES") specifying which
        molecule representation you plan to use.
    alphabet : list[str] | None, optional
        The alphabet to be used for the SMILES or SELFIES representation.
        It is common that the alphabet depends on the dataset used, so
        it is recommended to pass it as an argument. Default is None.
    max_sequence_length : int, optional
        The maximum length of the sequence. Default is infinity.
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
    **kwargs_for_oracle: dict
        Other keyword arguments to be passed to the oracle.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, oracle_name, string_representation="SMILES", force_isolation=False, batch_size=None, parallelize=False, num_workers=None, evaluation_budget=float("inf"), **kwargs_for_oracle
        Initializes a new instance of the abstract TDC class.

    References
    ----------
    [1] Artificial intelligence foundation for therapeutic science.
        Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
    """

    def __init__(
        self,
        oracle_name: str,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        alphabet: list[str] | None = None,
        max_sequence_length: int = np.inf,
        force_isolation: bool = False,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        **kwargs_for_oracle,
    ):
        if parallelize:
            print(
                "poli 🧪: TDCBlackBox parallelization is handled by oracles. Disabling it."
            )
            parallelize = False
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.oracle_name = oracle_name
        self.alphabet = alphabet
        self.max_sequence_length = max_sequence_length
        self.string_representation = string_representation

        from_smiles = string_representation.upper() == "SMILES"
        self.inner_function = get_inner_function(
            isolated_function_name="tdc__isolated",
            class_name="TDCIsolatedFunction",
            module_to_import="poli.core.chemistry.tdc_isolated_function",
            force_isolation=force_isolation,
            oracle_name=oracle_name,
            from_smiles=from_smiles,
            **kwargs_for_oracle,
        )

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        return self.inner_function(x, context)
