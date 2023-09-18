"""
This module implements the DDR3 docking task
using the TDC oracles [1].

[1] TODO: add reference.
"""
from typing import Tuple

import numpy as np

import selfies as sf

from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles

from poli.core.util.seeding import seed_numpy, seed_python


class DRD3BlackBox(TDCBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        from_smiles: bool = True,
    ):
        oracle_name = "3pbl_docking"
        super().__init__(
            oracle_name=oracle_name,
            info=info,
            batch_size=batch_size,
            from_smiles=from_smiles,
        )


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
        seed: int = None,
        batch_size: int = None,
        string_representation: str = "SMILES",
    ) -> Tuple[TDCBlackBox, np.ndarray, np.ndarray]:
        """
        TODO: document
        """
        # We start by seeding the RNGs
        seed_numpy(seed)
        seed_python(seed)

        # We check whether the string representation is valid
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
