import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import instance_function_as_isolated_process
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch


class GFPSelectionBlackBox(AbstractBlackBox):
    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        seed: int = None,
        force_isolation: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        if not force_isolation:
            try:
                from poli.objective_repository.gfp_select.isolated_function import (
                    GFPSelectIsolatedLogic,
                )

                self.inner_function = GFPSelectIsolatedLogic(seed=seed)
            except ImportError:
                self.inner_function = instance_function_as_isolated_process(
                    name="gfp_select__isolated", seed=seed
                )
        else:
            self.inner_function = instance_function_as_isolated_process(
                name="gfp_select__isolated", seed=seed
            )

    def _black_box(self, x: np.array, context=None) -> np.ndarray:
        """
        x is string sequence which we look-up in avilable df, return median Brightness
        """
        return self.inner_function(x, context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="gfp_select",
            max_sequence_length=237,  # max len of aaSequence
            aligned=True,  # TODO: perhaps add the fact that there is a random state here?
            fixed_length=True,
            deterministic=False,
            alphabet=AMINO_ACIDS,
            log_transform_recommended=False,
            discrete=True,
            fidelity=None,
            padding_token="",
        )


class GFPSelectionProblemFactory(AbstractProblemFactory):

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = GFPSelectionBlackBox(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            seed=seed,
            force_isolation=force_isolation,
        )
        x0 = f.inner_function.x0

        problem = Problem(f, x0)

        return problem
