"""
This is a registration script for the rdkit_logp problem,
whose black box objective function returns the log quotient
of solubility (a.k.a. logP) [1].

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

```
pip install rdkit selfies
```

[1] TODO: add reference
"""
from pathlib import Path
from typing import Tuple, List
import json

import numpy as np

from rdkit.Chem import Descriptors

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import strings_to_molecules


class LogPBlackBox(AbstractBlackBox):
    """
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
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        alphabet: List[str] = None,
        from_selfies: bool = False,
    ):
        if alphabet is None:
            # TODO: remove this as soon as we have a default alphabet
            assert info.alphabet is not None, (
                "We only support for the user to provide an alphabet (List[str]). "
                "Provide an alphabet in objective_function.create(...)"
            )
            alphabet = info.alphabet

        string_to_idx = {symbol: i for i, symbol in enumerate(alphabet)}

        self.string_to_idx = string_to_idx
        self.idx_to_string = {v: k for k, v in string_to_idx.items()}
        self.from_selfies = from_selfies
        self.from_smiles = not from_selfies

        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Assuming that x is an array of integers of length L,
        we use the alphabet to construct a SMILES string,
        and query QED from RDKit.
        """
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
        return ProblemSetupInformation(
            name="rdkit_logp",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = 0,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        path_to_alphabet: Path = None,
        alphabet: List[str] = None,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
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
        f = LogPBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
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
    logp_problem_factory = LogPProblemFactory()
    register_problem(
        logp_problem_factory,
        conda_environment_name="poli__chem",
        # force=True,
    )
