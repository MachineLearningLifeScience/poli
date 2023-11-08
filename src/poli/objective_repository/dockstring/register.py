"""
This script implements and registers a black box
objective function (and a repository) for dockstring [1].

[1] García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp, José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado. “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.” Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502. https://doi.org/10.1021/acs.jcim.1c01334.

"""
from typing import Tuple

import numpy as np

import selfies as sf

from dockstring import load_target

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import (
    translate_selfies_to_smiles,
    translate_smiles_to_selfies,
)

from poli.core.util.seeding import seed_numpy, seed_python


class DockstringBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        target_name: str = None,
        string_representation: str = "SMILES",
    ):
        assert (
            target_name is not None
        ), "Missing mandatory keyword argument 'target_name'. "

        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )
        self.target_name = target_name
        self.string_representation = string_representation

        self.target = load_target(target_name)

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Assuming that x is an array of strings (either in
        SMILES or in SELFIES representation).
        """
        assert len(x.shape) == 2, "Expected a 2D array of strings. "
        molecules_as_strings = ["".join(x_i) for x_i in x]

        if self.string_representation == "SELFIES":
            molecules_as_smiles = translate_selfies_to_smiles(molecules_as_strings)
        else:
            molecules_as_smiles = molecules_as_strings

        # TODO: Should we parallelize?
        scores = [self.target.dock(smiles)[0] for smiles in molecules_as_smiles]

        # Since our goal is maximization, and scores in dockstring
        # are better if they are lower, we return the negative of
        # the scores.
        return -np.array(scores).reshape(-1, 1)


class DockstringProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        # TODO: We might change this in the future for a
        # default dictionary, depending on whether we
        # are using SMILES or SELFIES.
        alphabet = None

        return ProblemSetupInformation(
            name="dockstring",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=alphabet,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        target_name: str = None,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        assert (
            target_name is not None
        ), "Missing mandatory keyword argument 'target_name'. "

        seed_numpy(seed)
        seed_python(seed)

        dockstring_black_box = DockstringBlackBox(
            info=self.get_setup_information(),
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            target_name=target_name,
            string_representation=string_representation,
        )

        # Using the initial example they provide in the
        # dockstring documentation (i.e. DRD2 and Risperidone)
        risperidone_smiles = "CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"
        if string_representation.upper() == "SMILES":
            # TODO: replace for proper smiles tokenization.
            x0 = np.array([list(risperidone_smiles)])
        elif string_representation.upper() == "SELFIES":
            risperidone_selfies = translate_smiles_to_selfies([risperidone_smiles])[0]
            risperidone_selfies_as_tokens = list(sf.split_selfies(risperidone_selfies))
            x0 = np.array([risperidone_selfies_as_tokens])
        else:
            raise ValueError(
                f"Invalid string representation. Expected SMILES or SELFIES but received {string_representation}."
            )

        return dockstring_black_box, x0, dockstring_black_box(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    dockstring_problem_factory = DockstringProblemFactory()
    register_problem(
        dockstring_problem_factory,
        conda_environment_name="poli__dockstring",
    )
