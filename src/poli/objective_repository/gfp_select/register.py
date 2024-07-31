import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import instance_function_as_isolated_process
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.gfp_select.information import gfp_select_info


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

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return gfp_select_info


class GFPSelectionProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        """
        The problem is set up such that all available sequences
        are provided in x0, however only batch_size amount of observations are known.
        I.e. f(x0[:batch_size]) is returned as f_0 .
        The task is to find the minimum, given that only limited inquiries (batch_size) can be done.
        Given that all X are known it is recommended to use an acquisition function to rank
        and inquire the highest rated sequences with the _black_box.
        """
        return gfp_select_info

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


if __name__ == "__main__":
    from poli.core.registry import register_problem

    gfp_problem_factory = GFPSelectionProblemFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__protein_cbas",
    )
