"""
This is a registration script for the rdkit_qed problem,
whose black box objective function returns the quantitative
estimate of druglikeness, which is a continuous version
of Lipinsky's rule of 5 [1].

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

```
pip install rdkit selfies
```

[1] TODO: add reference
"""
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np

from rdkit import Chem
from rdkit.Chem.QED import qed

import selfies as sf

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import string_to_molecule


class QEDBlackBox(AbstractBlackBox):
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
        L: int = np.inf,
        alphabet: Dict[str, int] = None,
        from_selfies: bool = False,
    ):
        if alphabet is None:
            raise ValueError("Alphabet must be provided to the QEDBlackBox.")
        self.alphabet = alphabet
        self.inverse_alphabet = {v: k for k, v in alphabet.items()}
        self.from_selfies = from_selfies
        self.from_smiles = not from_selfies

        super().__init__(L=L)

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Assuming that x is an array of integers of length L,
        we use the alphabet to construct a SMILES string,
        and query QED from RDKit.
        """
        molecule_string = "".join(
            [
                self.inverse_alphabet[i]
                for i in x.flatten()
                if self.inverse_alphabet[i] not in ["[unk]", "[unk2]"]
            ]
        )
        try:
            molecule = string_to_molecule(
                molecule_string, from_selfies=self.from_selfies
            )
        except ValueError:
            # If the molecule cannot be parsed, return NaN
            return np.array([[np.nan]])

        qed_value = qed(molecule)

        # If the qed value is not a float, return NaN
        if not isinstance(qed_value, float):
            return np.array([[np.nan]])

        return np.array([[qed_value]])


class QEDProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(
            name="rdkit_qed",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = 0,
        alphabet: Dict[str, int] = None,
        path_to_alphabet: Path = None,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        if path_to_alphabet is None and alphabet is None:
            # TODO: add support for more file types
            raise ValueError(
                "Missing required keyword argument: either path_to_alphabet or alphabet must be provided. \n"
                "- alphabet could be a Dict[str, int], or \n "
                "- path_to_alphabet could be the Path to a json file {str: int} \n "
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

        L = self.get_setup_information().get_max_sequence_length()
        f = QEDBlackBox(
            L=L,
            alphabet=self.alphabet,
            from_selfies=string_representation.upper() == "SELFIES",
        )

        # The sequence "C"
        if string_representation.upper() == "SMILES":
            x0 = np.array([[self.alphabet["C"]]])
        else:
            x0 = np.array([[self.alphabet["[C]"]]])

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
