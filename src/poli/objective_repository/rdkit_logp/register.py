"""Registers the logP objective factory and black-box

This is a registration script for the rdkit_logp problem,
whose black box objective function returns the log quotient
of solubility (a.k.a. logP).

This black box is a simple wrapper around RDKit's
descriptors. We allow for both SMILES and SELFIES
strings.

The problem is registered as 'rdkit_logp', and it uses
a conda environment called 'poli__chem' (see the
environment.yml file in this folder). If you want to
run it locally without creating a new environemnt,
these are the extra requirements:

- rdkit
- selfies

If you are interested in running this directly,
instead of inside an isolated process, run:

`
pip install rdkit selfies
`
"""

from pathlib import Path
from typing import Tuple, List, Literal
import json

import numpy as np

from rdkit.Chem import Descriptors

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

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
    info : ProblemSetupInformation
        The problem setup information.
    batch_size : int, optional
        The batch size for processing multiple inputs simultaneously, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation, by default False.
    num_workers : int, optional
        The number of workers to use for parallel computation, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    from_selfies : bool, optional
        Flag indicating whether the input is a SELFIES string,
        by default False (i.e. we expect a SMILES string).

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
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        from_selfies: bool = False,
    ):
        """
        Initializes the LogP black box.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information.
        batch_size : int, optional
            The batch size for processing multiple inputs simultaneously, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the computation, by default False.
        num_workers : int, optional
            The number of workers to use for parallel computation, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        from_selfies : bool, optional
            Flag indicating whether the input is a SELFIES string,
            by default False (i.e. we expect a SMILES string).
        """
        self.from_selfies = from_selfies
        self.from_smiles = not from_selfies

        super().__init__(
            info=info,
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


class LogPProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        """
        Returns the setup information for the logP problem.

        Returns
        -------
        info : ProblemSetupInformation
            The setup information for the logP problem.
        """
        return ProblemSetupInformation(
            name="rdkit_logp",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[LogPBlackBox, np.ndarray, np.ndarray]:
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
        f : LogPBlackBox
            The logP black box function.
        x0 : np.ndarray
            The initial input values (a single carbon).
        y0 : np.ndarray
            The initial output values (the logP of a single carbon).
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        problem_info = self.get_setup_information()
        f = LogPBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            from_selfies=string_representation.upper() == "SELFIES",
        )

        # The sequence "C"
        if string_representation.upper() == "SMILES":
            x0 = np.array([["C"]])
        else:
            x0 = np.array([["[C]"]])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    logp_problem_factory = LogPProblemFactory()
    register_problem(
        logp_problem_factory,
        conda_environment_name="poli__chem",
        # force=True,
    )
