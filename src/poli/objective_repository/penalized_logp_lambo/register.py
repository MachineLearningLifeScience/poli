"""
This module implements LogP in _exactly_ the same was
as LaMBO does it [1]. We do this by importing the function
they use to compute `logp`.

References
----------
[1] “Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders.”
Stanton, Samuel, Wesley Maddox, Nate Gruver, Phillip Maffettone,
Emily Delaney, Peyton Greenside, and Andrew Gordon Wilson.
arXiv, July 12, 2022. http://arxiv.org/abs/2203.12742.
"""
from typing import Tuple
import numpy as np

from lambo.tasks.chem.logp import logP

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles

from poli.core.util.seeding import seed_numpy, seed_python


class PenalizedLogPLamboBlackBox(AbstractBlackBox):
    """
    A black box objective function that returns the
    penalized logP of a molecule, using the same function
    that LaMBO [1] uses.

    In particular, they adjust the penalized logP using
    some "magic numbers", which are the empirical means
    and standard deviations of the dataset.
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        from_smiles: bool = True,
        penalized: bool = True,
    ):
        """
        TODO: document
        """
        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.from_smiles = from_smiles
        self.penalized = penalized

    def _black_box(self, x: np.ndarray, context: dict = None):
        """
        Assuming that x is an array of strings (of shape [b,L]),
        we concatenate, translate to smiles if it's
        necessary, and then computes the penalized logP.

        If the inputs are SELFIES, it translates first to SMILES,
        and then computes the penalized logP. If the translation
        threw an error, we return NaN instead.
        """
        if not x.dtype.kind in ["U", "S"]:
            raise ValueError(
                f"We expect x to be an array of strings, but we got {x.dtype}"
            )

        molecule_strings = ["".join([x_ij for x_ij in x_i.flatten()]) for x_i in x]

        if not self.from_smiles:
            molecule_strings = translate_selfies_to_smiles(molecule_strings)

        logp_scores = []
        for molecule_string in molecule_strings:
            if molecule_string is None:
                logp_scores.append(np.nan)
            else:
                logp_scores.append(logP(molecule_string, penalized=self.penalized))

        return np.array(logp_scores).reshape(-1, 1)


class PenalizedLogPLamboProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        # TODO: do they have an alphabet?
        return ProblemSetupInformation(
            name="penalized_logp_lambo",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        penalized: bool = True,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        seed_numpy(seed)
        seed_python(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        problem_info = self.get_setup_information()
        f = PenalizedLogPLamboBlackBox(
            problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            from_smiles=(string_representation.upper() == "SMILES"),
            penalized=penalized,
        )

        if string_representation.upper() == "SMILES":
            x0 = np.array([["C"]])
        else:
            x0 = np.array([["[C]"]])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        PenalizedLogPLamboProblemFactory(),
        conda_environment_name="poli__lambo",
    )
