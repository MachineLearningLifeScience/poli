"""RFP objective factory and black box function."""

__author__ = "Simon Bartels"

from pathlib import Path

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import FoldXNotFoundException
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import instance_function_as_isolated_process
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.foldx_rfp_lambo import CORRECT_SEQ, PROBLEM_SEQ
from poli.objective_repository.foldx_rfp_lambo.information import AMINO_ACIDS


class FoldXRFPLamboBlackBox(AbstractBlackBox):
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

        if not (Path.home() / "foldx" / "foldx").exists():
            raise FoldXNotFoundException(
                "FoldX wasn't found in ~/foldx/foldx. Please install it."
            )
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

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="foldx_rfp_lambo",
            max_sequence_length=244,
            aligned=False,
            fixed_length=False,
            deterministic=True,  # ?
            alphabet=AMINO_ACIDS,
            discrete=True,
            fidelity=None,
            padding_token="-",
        )


class FoldXRFPLamboProblemFactory(AbstractProblemFactory):
    def __init__(self):
        self.alphabet = AMINO_ACIDS
        self.problem_sequence = PROBLEM_SEQ
        self.correct_sequence = CORRECT_SEQ

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a FoldX-RFP Lambo-specific black box instance, alongside initial
        observations.

        Parameters
        ----------
        seed : int, optional
            The seed value for random number generation, by default None.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The evaluation budget, by default float("inf").
        force_isolation : bool, optional
            Should the problem
        Returns
        -------
        f : RaspBlackBox
            The FoldXRFP black box instance.
        x0 : np.ndarray
            The initial observations.
        y0 : np.ndarray
            The initial observations (i.e. the stability of the wildtypes).
        """
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

        black_box = FoldXRFPLamboBlackBox(
            seed=seed,
            parallelize=parallelize,
            num_workers=num_workers,
            batch_size=batch_size,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        x0 = black_box.inner_function.x0

        return Problem(black_box, x0)
