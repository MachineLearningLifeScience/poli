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

from typing import Literal, Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch


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
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        penalized: bool = True,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        from_smiles = string_representation.upper() == "SMILES"
        self.from_smiles = from_smiles
        self.penalized = penalized
        self.force_isolation = force_isolation

        # Testing whether we can import in isolation
        # to eagerly throw any import errors
        _ = get_inner_function(
            isolated_function_name="penalized_logp_lambo__isolated",
            class_name="PenalizedLogPIsolatedLogic",
            module_to_import="poli.objective_repository.penalized_logp_lambo.isolated_function",
            force_isolation=force_isolation,
            from_smiles=from_smiles,
            penalized=penalized,
        )

    def _black_box(self, x: np.ndarray, context: dict = None):
        """
        Assuming that x is an array of strings (of shape [b,L]),
        we concatenate, translate to smiles if it's
        necessary, and then computes the penalized logP.

        If the inputs are SELFIES, it translates first to SMILES,
        and then computes the penalized logP. If the translation
        threw an error, we return NaN instead.
        """
        inner_function = get_inner_function(
            isolated_function_name="penalized_logp_lambo__isolated",
            class_name="PenalizedLogPIsolatedLogic",
            module_to_import="poli.objective_repository.penalized_logp_lambo.isolated_function",
            force_isolation=self.force_isolation,
            from_smiles=self.from_smiles,
            penalized=self.penalized,
        )
        return inner_function(x, context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="penalized_logp_lambo",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            alphabet=None,  # TODO: add when we settle for an alphabet
            deterministic=True,
            log_transform_recommended=False,
            discrete=True,
            fidelity=None,
            padding_token="",
        )


class PenalizedLogPLamboProblemFactory(AbstractProblemFactory):
    def create(
        self,
        penalized: bool = True,
        string_representation: str = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        f = PenalizedLogPLamboBlackBox(
            string_representation=string_representation.upper(),
            penalized=penalized,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        if string_representation.upper() == "SMILES":
            x0 = np.array([["C"]])
        else:
            x0 = np.array([["[C]"]])

        problem = Problem(f, x0)

        return problem
