"""Registers the black box factory and function for QED using RDKit.

This is a registration script for the rdkit_qed problem,
whose black box objective function returns the quantitative
estimate of druglikeness, which is a continuous version
of Lipinsky's rule of 5.

This black box is a simple wrapper around RDKit's
Chem.QED.qed function, which returns a float between
0 and 1. We allow for both SMILES and SELFIES strings.
"""

from typing import Literal

import numpy as np
from rdkit.Chem.QED import qed

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.chemistry.string_to_molecule import strings_to_molecules
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.rdkit_qed.information import rdkit_qed_info


class QEDBlackBox(AbstractBlackBox):
    """Quantitative estimate of druglikeness (QED) black box.

    A simple black box that returns the QED
    of a molecule. By default, we assume that the
    result of concatenating the tokens will be
    a SMILES string, but you can set the context
    variable "from_selfies" to True to indicate
    that the input is a SELFIES string.

    RDKit's Chem.MolFromSmiles function and qed are known
    for failing silently, so we return NaN if the
    molecule cannot be parsed or if qed returns
    something other than a float.

    Parameters
    ----------
    string_representation : Literal["SMILES", "SELFIES"], optional
        The string representation to use, by default "SMILES".
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
        it computes the qed of the molecule in x.
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        """
        Initialize the QEDBlackBox.

        Parameters
        ----------
        string_representation : Literal["SMILES", "SELFIES"], optional
            The string representation to use, by default "SMILES".
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
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        assert string_representation.upper() in ["SMILES", "SELFIES"]
        self.from_selfies = string_representation.upper() == "SELFIES"
        self.from_smiles = string_representation.upper() == "SMILES"

        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """Computes the qed of the molecule in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings or integers.
            If the elements are integers, they are converted to strings using the alphabet.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            Array of shape [b, 1] containing the qed of the molecule in x.
        """
        # We check if the user provided an array of strings
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

        # Computes the QED values for each molecule.
        qed_values = []
        for molecule in molecules:
            if molecule is not None:
                qed_value = qed(molecule)

                # If the qed value is not a float, return NaN
                if not isinstance(qed_value, float):
                    qed_value = np.nan

            # If the molecule is None, then RDKit failed
            # to parse it, and we return NaN.
            else:
                qed_value = np.nan

            qed_values.append(qed_value)

        return np.array(qed_values).reshape(-1, 1)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        """Returns the black box information for the QED problem.

        Returns
        -------
        BlackBoxInformation
            The black box information for the QED problem.
        """
        return rdkit_qed_info


class QEDProblemFactory(AbstractProblemFactory):
    """Problem factory for the QED problem.

    The Quantitative Estimate of Druglikeness (QED) objective
    function returns a "continuous" version of Lipinsky's rule
    of 5, which is a heuristic to evaluate the druglikeness
    of a molecule (discarding molecules that are e.g. too heavy).

    Methods
    -------
    get_setup_information()
        Returns the setup information for the problem.
    create(...)
        Creates a problem instance with the specified parameters.
    """

    def get_setup_information(self) -> BlackBoxInformation:
        return rdkit_qed_info

    def create(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Problem:
        """Creates a QED black box function and initial observations.

        Parameters
        ----------
        string_representation : str, optional
            The string representation to use, by default "SMILES".
            It must be either "SMILES" or "SELFIES".
        seed : int, optional
            The seed value for random number generation, by default None.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The maximum number of evaluations, by default float("inf").

        Returns
        -------
        problem : Problem
            The QED problem instance, containing the black box and the initial input.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        f = QEDBlackBox(
            string_representation=string_representation.upper(),
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # The sequence "C"
        if string_representation.upper() == "SMILES":
            x0 = np.array([["C"]])
        else:
            x0 = np.array([["[C]"]])

        return Problem(f, x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    qed_problem_factory = QEDProblemFactory()
    register_problem(
        qed_problem_factory,
        conda_environment_name="poli__chem",
        # force=True,
    )
