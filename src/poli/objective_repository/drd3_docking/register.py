"""
This module implements the DDR3 docking task
using the TDC oracles [1].

[1] TODO: add reference.
"""
from typing import Tuple

import numpy as np

from tdc import Oracle

import selfies as sf

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles


class DRD3BlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        from_smiles: bool = True,
    ):
        super().__init__(info, batch_size)
        self.oracle = Oracle(name="3pbl_docking")
        self.from_smiles = from_smiles

    def _black_box(self, x, context=None):
        """
        Assuming x is an array of strings,
        we concatenate them and then
        compute the docking score.
        """
        if not x.dtype.kind in ["U", "S"]:
            raise ValueError(
                f"We expect x to be an array of strings, but we got {x.dtype}"
            )

        molecule_strings = ["".join([x_ij for x_ij in x_i.flatten()]) for x_i in x]

        if not self.from_smiles:
            molecule_strings = translate_selfies_to_smiles(molecule_strings)

        docking_scores = []
        for molecule_string in molecule_strings:
            docking_scores.append(self.oracle(molecule_string))

        return np.array(docking_scores).reshape(-1, 1)


class DRD3ProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(
            name="drd3_docking",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = 0,
        batch_size: int = None,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        problem_info = self.get_setup_information()
        f = DRD3BlackBox(
            info=problem_info,
            batch_size=batch_size,
            from_smiles=string_representation.upper() == "SMILES",
        )

        # Initial example (from the TDC docs)
        x0_smiles = "c1ccccc1"
        x0_selfies = translate_selfies_to_smiles([x0_smiles])[0]

        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        DRD3ProblemFactory(),
        conda_environment_name="poli__lambo",
        force=True,
    )
