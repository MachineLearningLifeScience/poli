"""RFP objective factory and black box function."""

__author__ = "Simon Bartels"

import logging
from pathlib import Path
import os
import random
from typing import Tuple

import numpy as np

# import hydra
# import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem import Problem
from poli.core.black_box_information import BlackBoxInformation

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.objective_repository.foldx_rfp_lambo import PROBLEM_SEQ, CORRECT_SEQ
from poli.core.util.isolation.instancing import instance_function_as_isolated_process
from poli.core.util.seeding import seed_python_numpy_and_torch

from poli.objective_repository.foldx_rfp_lambo.information import (
    foldx_rfp_lambo_information,
    AMINO_ACIDS,
)


class FoldxRFPLamboBlackBox(AbstractBlackBox):
    def __init__(
        self,
        seed: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        batch_size: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.inverse_alphabet = {i + 1: AMINO_ACIDS[i] for i in range(len(AMINO_ACIDS))}
        self.inverse_alphabet[0] = "-"

        if not force_isolation:
            try:
                from poli.objective_repository.foldx_rfp_lambo.isolated_function import (
                    RFPWrapperIsolatedLogic,
                )

                self.inner_function = RFPWrapperIsolatedLogic(seed=seed)
            except (ImportError, FileNotFoundError):
                self.inner_function = instance_function_as_isolated_process(
                    name="foldx_rfp_lambo__isolated",
                )
        else:
            self.inner_function = instance_function_as_isolated_process(
                name="foldx_rfp_lambo__isolated",
            )

    def _black_box(self, x, context=None):
        return self.inner_function(x, context)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return foldx_rfp_lambo_information


class FoldxRFPLamboProblemFactory(AbstractProblemFactory):
    def __init__(self):
        self.alphabet = AMINO_ACIDS
        self.problem_sequence = PROBLEM_SEQ
        self.correct_sequence = CORRECT_SEQ

    def get_setup_information(self) -> BlackBoxInformation:
        return foldx_rfp_lambo_information

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Problem:
        # TODO: allow for bigger batch_sizes
        # For now (and because of the way the black box is implemented)
        # we only allow for batch_size=1
        if batch_size is None:
            batch_size = 1
        else:
            assert batch_size == 1

        # make problem reproducible
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        black_box = FoldxRFPLamboBlackBox(
            seed=seed,
            parallelize=parallelize,
            num_workers=num_workers,
            batch_size=batch_size,
            evaluation_budget=evaluation_budget,
        )
        x0 = black_box.inner_function.x0

        return Problem(black_box, x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    rfp_problem_factory = FoldxRFPLamboProblemFactory()
    register_problem(
        rfp_problem_factory,
        conda_environment_name="poli__lambo",
        force=True,
    )
