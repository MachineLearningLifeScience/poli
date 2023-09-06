"""
In this module, we implement a synthetic-accessibility 
objective using the TDC oracles [1].

[1] TODO: add reference.
"""
from typing import Tuple

import numpy as np

import selfies as sf

from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles


class SABlackBox(TDCBlackBox):
    def __init__(self, info: ProblemSetupInformation, batch_size: int = None, from_smiles: bool = True):
        oracle_name = "SA"
        super().__init__(oracle_name, info, batch_size, from_smiles)

class SAProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(
            name="sa_tdc",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = 0,
        batch_size: int = None,
        string_representation: str = "SMILES",
    ) -> Tuple[SABlackBox, np.ndarray, np.ndarray]:
        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        problem_info = self.get_setup_information()
        f = SABlackBox(
            info=problem_info,
            batch_size=batch_size,
            from_smiles=string_representation.upper() == "SMILES",
        )

        # Initial example (from the TDC docs)
        x0_smiles = "CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1"
        x0_selfies = translate_selfies_to_smiles([x0_smiles])[0]

        # TODO: change for proper tokenization in the SMILES case.
        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        SAProblemFactory(),
        conda_environment_name="poli__lambo",
        force=True,
    )
