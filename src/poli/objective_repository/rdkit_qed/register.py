"""Registers the black box factory and function for QED using RDKit.

This is a registration script for the rdkit_qed problem,
whose black box objective function returns the quantitative
estimate of druglikeness, which is a continuous version
of Lipinsky's rule of 5.

This black box is a simple wrapper around RDKit's
Chem.QED.qed function, which returns a float between
0 and 1. We allow for both SMILES and SELFIES strings.

The problem is registered as 'rdkit_qed', and it uses
a conda environment called 'poli__chem' (see the
environment.yml file in this folder). If you want to
run it locally without creating a new environemnt,
these are the extra requirements:

- rdkit
- selfies

Run:

`
pip install rdkit selfies
`
"""
from pathlib import Path
from typing import Tuple, List, Literal
import json

import numpy as np

from rdkit.Chem.QED import qed

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import strings_to_molecules

from poli.core.util.seeding import seed_python_numpy_and_torch


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
    alphabet : List[str], optional
        The alphabet to use for the black box. If not provided,
        the alphabet from the problem setup information will be used.
        We strongly advice providing an alphabet.
    from_selfies : bool, optional
        Flag indicating whether the input is a SELFIES string, by default False.

    Attributes
    ----------
    alphabet : List[str]
        The alphabet to use for the black box.
    string_to_idx : dict
        The mapping of symbols to their corresponding indices in the alphabet.
    idx_to_string : dict
        The mapping of indices to their corresponding symbols in the alphabet.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the qed of the molecule in x.
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        alphabet: List[str] = None,
        from_selfies: bool = False,
    ):
        """
        Initialize the QEDBlackBox.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information object.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The maximum number of evaluations, by default float("inf").
        alphabet : List[str], optional
            The alphabet for encoding molecules, by default it's
            the one inside the problem setup information. We strongly
            advice providing an alphabet.
        from_selfies : bool, optional
            Flag indicating whether the molecules are encoded using SELFIES, by default False.
        """
        super().__init__(info, batch_size, parallelize, num_workers)
        if alphabet is None:
            # TODO: remove this as soon as we have a default alphabet
            assert info.alphabet is not None, (
                "We only support for the user to provide an alphabet (List[str]). "
                "Provide an alphabet in objective_function.create(...)"
            )
            alphabet = info.alphabet

        string_to_idx = {symbol: i for i, symbol in enumerate(alphabet)}

        self.alphabet = alphabet
        self.string_to_idx = string_to_idx
        self.idx_to_string = {v: k for k, v in string_to_idx.items()}
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
        elif x.dtype.kind in ["i", "f"]:
            molecule_strings = [
                "".join([self.idx_to_string[x_ij] for x_ij in x_i.flatten()])
                for x_i in x
            ]
        else:
            raise ValueError(
                f"Unsupported dtype: {x.dtype}. "
                "The input must be an array of strings or integers."
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

    def get_setup_information(self) -> ProblemSetupInformation:
        # TODO: Add a default alphabet here.
        return ProblemSetupInformation(
            name="rdkit_qed",
            max_sequence_length=np.inf,
            aligned=True,
            alphabet=None,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        alphabet: List[str] = None,
        path_to_alphabet: Path = None,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
    ) -> Tuple[QEDBlackBox, np.ndarray, np.ndarray]:
        """Creates a QED black box function and initial observations.

        Parameters
        ----------
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
        alphabet : List[str], optional
            The alphabet to use for encoding molecules, by default None.
            If not provided, the alphabet from the problem setup information
            will be used. We strongly advice providing an alphabet.
        path_to_alphabet : Path, optional
            The path to a json file containing the alphabet, by default None.
            If not provided, the alphabet from the problem setup information
            will be used. We strongly advice providing an alphabet.
        string_representation : str, optional
            The string representation to use, by default "SMILES".
            It must be either "SMILES" or "SELFIES".

        Returns
        -------
        f : QEDBlackBox
            The QED black box function.
        x0 : np.ndarray
            The initial input (a single carbon).
        y0 : np.ndarray
            The initial output (the qed of a single carbon).
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if path_to_alphabet is None and alphabet is None:
            # TODO: add support for more file types
            raise ValueError(
                "Missing required keyword argument: either path_to_alphabet or alphabet must be provided. \n"
                "- alphabet could be a List[str], or \n "
                "- path_to_alphabet could be the Path to a json file [token_1, token_2, ...] \n "
            )

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        if alphabet is None:
            if isinstance(path_to_alphabet, str):
                path_to_alphabet = Path(path_to_alphabet.strip()).resolve()

            if not path_to_alphabet.exists():
                raise ValueError(f"Path to alphabet {path_to_alphabet} does not exist.")

            if not path_to_alphabet.suffix == ".json":
                # TODO: add support for more file types
                raise ValueError(
                    f"Path to alphabet {path_to_alphabet} must be a json file."
                )

            with open(path_to_alphabet, "r") as f:
                alphabet = json.load(f)

        self.alphabet = alphabet

        problem_info = self.get_setup_information()
        f = QEDBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            alphabet=self.alphabet,
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
    qed_problem_factory = QEDProblemFactory()
    register_problem(
        qed_problem_factory,
        conda_environment_name="poli__chem",
        # force=True,
    )
