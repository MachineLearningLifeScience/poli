"""Registers the logP objective factory and black-box

This is a registration script for the rdkit_logp problem,
whose black box objective function returns the log quotient
of solubility (a.k.a. logP).

This black box is a simple wrapper around RDKit's
descriptors. We allow for both SMILES and SELFIES
strings.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from rdkit.Chem import Descriptors

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.chemistry.tdc_problem import TDCProblem
from poli.core.problem import Problem
from poli.core.util.chemistry.string_to_molecule import strings_to_molecules
from poli.core.util.seeding import seed_python_numpy_and_torch


class LogPBlackBox(AbstractBlackBox):
    """Log-solubility of a small molecule.

    A simple black box that returns the LogP
    of a molecule. By default, we assume that the
    result of concatenating the tokens will be
    a SMILES string, but you can set the context
    variable "from_selfies" to True to indicate
    that the input is a SELFIES string.

    RDKit's Chem.MolFromSmiles function and logP are known
    for failing silently, so we return NaN if the
    molecule cannot be parsed or if qed returns
    something other than a float.

    Parameters
    ----------
    string_representation : Literal["SMILES", "SELFIES"], optional
        The string representation to use, by default "SMILES".
    alphabet : list[str] | None, optional
        The alphabet to be used for the SMILES or SELFIES representation.
        It is common that the alphabet depends on the dataset used, so
        it is recommended to pass it as an argument. Default is None.
    max_sequence_length : int, optional
        The maximum length of the sequence. Default is infinity.
    batch_size : int, optional
        The batch size for processing multiple inputs simultaneously, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation, by default False.
    num_workers : int, optional
        The number of workers to use for parallel computation, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Attributes
    ----------
    from_selfies : bool
        Flag indicating whether the input is a SELFIES string.
    from_smiles : bool
        Flag indicating whether the input is a SMILES string.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the logP of the molecule in x.
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
        Initializes the LogP black box.

        Parameters
        ----------
        string_representation : Literal["SMILES", "SELFIES"], optional
            The string representation to use, by default "SMILES".
        alphabet : list[str] | None, optional
            The alphabet to be used for the SMILES or SELFIES representation.
            It is common that the alphabet depends on the dataset used, so
            it is recommended to pass it as an argument. Default is None.
        max_sequence_length : int, optional
            The maximum length of the sequence. Default is infinity.
        batch_size : int, optional
            The batch size for processing multiple inputs simultaneously, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the computation, by default False.
        num_workers : int, optional
            The number of workers to use for parallel computation, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        """
        assert string_representation.upper() in ["SMILES", "SELFIES"]
        self.from_selfies = string_representation.upper() == "SELFIES"
        self.from_smiles = string_representation.upper() == "SMILES"
        self.alphabet = alphabet
        self.max_sequence_length = max_sequence_length
        self.string_representation = string_representation

        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """Computes the logP of a molecule x (array of strings).

        Assuming that x is an array of integers of length L,
        we use the alphabet to construct a SMILES string,
        and query logP from RDKit.
        """
        if x.dtype.kind in ["U", "S"]:
            molecule_strings = ["".join([x_ij for x_ij in x_i.flatten()]) for x_i in x]
        else:
            raise ValueError(
                f"Unsupported dtype: {x.dtype}. "
                "The input must be an array of strings."
            )

        # Transforms strings into RDKit molecules.
        # Those that cannot be parsed are set to None.
        molecules = strings_to_molecules(
            molecule_strings, from_selfies=self.from_selfies
        )

        # Computes the LogP values for each molecule.
        logp_values = []

        for molecule in molecules:
            if molecule is not None:
                logp_value = Descriptors.MolLogP(molecule)

                # If the qed value is not a float, return NaN
                if not isinstance(logp_value, float):
                    logp_value = np.nan

            # If the molecule is None, then RDKit failed
            # to parse it, and we return NaN.
            else:
                logp_value = np.nan

            logp_values.append(logp_value)

        return np.array(logp_values).reshape(-1, 1)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="rdkit_logp",
            max_sequence_length=self.max_sequence_length,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=self.alphabet,  # TODO: add once we settle for one
            log_transform_recommended=False,
            discrete=True,
            fidelity=None,
            padding_token="",
        )


class LogPProblemFactory(AbstractProblemFactory):
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
        """Creates a logP problem instance.

        Parameters
        ----------
        string_representation : str, optional
            The string representation of the input, by default "SMILES".
            Supported values are "SMILES" and "SELFIES".
        seed : int, optional
            The seed value for random number generation, by default None.
        batch_size : int, optional
            The batch size for processing multiple inputs simultaneously, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the computation, by default False.
        num_workers : int, optional
            The number of workers to use for parallel computation, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem : Problem
            The logP problem instance, containing the black box and the initial input.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        self.string_representation = string_representation

        f = LogPBlackBox(
            string_representation=string_representation.upper(),
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # The sequence "C"
        if string_representation.upper() == "SMILES":
            x0 = np.array([["C" * 10]])
        else:
            x0 = np.array([["[C]" * 10]])

        problem = TDCProblem(f, x0)

        return problem
