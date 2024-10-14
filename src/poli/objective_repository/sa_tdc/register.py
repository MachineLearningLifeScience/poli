"""
Implements a synthetic-accessibility objective using the TDC oracles [1].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import selfies as sf

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.chemistry.tdc_problem import TDCProblem
from poli.core.problem import Problem
from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies
from poli.core.util.seeding import seed_python_numpy_and_torch


class SABlackBox(TDCBlackBox):
    """Synthetic-accessibility black box implementation using the TDC oracles [1].

    Parameters
    ----------
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
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        alphabet: list[str] | None = None,
        max_sequence_length: int = np.inf,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        """
        Initialize the SABlackBox object.

        Parameters
        ----------
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
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The maximum number of evaluations, by default float("inf").
        """
        super().__init__(
            oracle_name="SA",
            string_representation=string_representation,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            force_isolation=force_isolation,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="sa_tdc",
            max_sequence_length=self.max_sequence_length,
            aligned=False,
            fixed_length=False,
            deterministic=True,  # ?
            alphabet=self.alphabet,  # TODO: add alphabet once we settle for one for SMLIES/SELFIES.
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class SAProblemFactory(AbstractProblemFactory):
    """Problem factory for the synthetic-accessibility problem.

    Methods
    -------
    create(...)
        Creates a synthetic-accessibility problem instance with the specified parameters.
    """

    def create(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        alphabet: list[str] | None = None,
        max_sequence_length: int = np.inf,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a synthetic-accessibility problem instance with the specified parameters.

        Parameters
        -----------
        string_representation : str, optional
            The string representation of the input molecules. Default is "SMILES".
        alphabet : list[str] | None, optional
            The alphabet to be used for the SMILES or SELFIES representation.
            It is common that the alphabet depends on the dataset used, so
            it is recommended to pass it as an argument. Default is None.
        max_sequence_length : int, optional
            The maximum length of the sequence. Default is infinity.
        seed:  int, optional
            The seed for random number generation. Default is None.
        batch_size:  int, optional
            The batch size for simultaneous evaluation. Default is None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation. Default is False.
        num_workers:  int, optional
            The number of workers for parallel evaluation. Default is None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        force_isolation: bool, optional
            Flag indicating whether to force isolation inside the
            black box. Default is False.

        Returns
        --------
        f: SABlackBox
            The synthetic-accessibility black box function.
        x0: np.ndarray
            The initial input (taken from TDC: CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1).
        y0: np.ndarray
            The initial output (i.e. the corresponding SA).
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        f = SABlackBox(
            string_representation=string_representation,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Initial example (from the TDC docs)
        x0_smiles = "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1"
        x0_selfies = translate_smiles_to_selfies([x0_smiles])[0]

        # TODO: change for proper tokenization in the SMILES case.
        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        problem = TDCProblem(
            black_box=f,
            x0=x0,
        )
        return problem
