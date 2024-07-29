from typing import Literal
from warnings import warn

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.gfp_cbas.information import gfp_cbas_info


class GFPCBasBlackBox(AbstractBlackBox):
    def __init__(
        self,
        problem_type: Literal["gp", "vae", "elbo"],
        functional_only: bool = False,
        ignore_stops: bool = True,
        unique=True,
        n_starting_points: int = 1,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        seed: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        negate: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.problem_type = problem_type
        self.functional_only = functional_only
        self.ignore_stops = ignore_stops
        self.unique = unique
        self.force_isolation = force_isolation
        self.n_starting_points = n_starting_points
        self.seed = seed
        self.negate = negate

        inner_function = get_inner_function(
            isolated_function_name="gfp_cbas__isolated",
            class_name="GFPCBasIsolatedLogic",
            module_to_import="poli.objective_repository.gfp_cbas.isolated_function",
            seed=self.seed,
            force_isolation=self.force_isolation,
            quiet=False,
            problem_type=self.problem_type,
            info=gfp_cbas_info,
            n_starting_points=self.n_starting_points,
            functional_only=self.functional_only,
            ignore_stops=self.ignore_stops,
            unique=self.unique,
        )
        self.x0 = inner_function.x0

    def _black_box(self, x: np.array, context=None) -> np.ndarray:
        """
        x is encoded sequence return function value given problem name
        """
        inner_function = get_inner_function(
            isolated_function_name="gfp_cbas__isolated",
            class_name="GFPCBasIsolatedLogic",
            module_to_import="poli.objective_repository.gfp_cbas.isolated_function",
            seed=self.seed,
            force_isolation=self.force_isolation,
            quiet=True,
            problem_type=self.problem_type,
            info=gfp_cbas_info,
            n_starting_points=self.n_starting_points,
            functional_only=self.functional_only,
            ignore_stops=self.ignore_stops,
            unique=self.unique,
        )
        if self.negate:
            return -inner_function(x, context=context)
        return inner_function(x, context=context)

    def __iter__(self, *args, **kwargs):
        warn(f"{self.__class__.__name__} iteration invoked. Not implemented!")

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return gfp_cbas_info


class GFPCBasProblemFactory(AbstractProblemFactory):
    def __init__(self, problem_type: str = "gp") -> None:
        super().__init__()
        if problem_type.lower() not in ["gp", "vae", "elbo"]:
            raise NotImplementedError(
                f"Specified problem type: {problem_type} does not exist!"
            )
        self.problem_type = problem_type.lower()

    def get_setup_information(self) -> BlackBoxInformation:
        """
        The problem is set up such that all available sequences
        are provided in x0, however only batch_size amount of observations are known.
        I.e. f(x0[:batch_size]) is returned as f_0 .
        The task is to find the minimum, given that only limited inquiries (batch_size) can be done.
        Given that all X are known it is recommended to use an acquisition function to rank
        and inquire the highest rated sequences with the _black_box.
        """
        return gfp_cbas_info

    def create(
        self,
        problem_type: Literal["gp", "vae", "elbo"] = "gp",
        n_starting_points: int = 1,
        functional_only: bool = False,
        unique: bool = True,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        negate: bool = False,
    ) -> Problem:
        """
        Seed value required to shuffle the data, otherwise CSV asset data index unchanged.
        We optimize with respect to one GFP WT sequence by default.
        If more starting points are requested the sequences are provided at random.
        """
        self.problem_type = problem_type
        if seed is not None:
            seed_python_numpy_and_torch(seed)
        problem_info = self.get_setup_information()
        f = GFPCBasBlackBox(
            problem_type=problem_type,
            functional_only=functional_only,
            unique=unique,
            n_starting_points=n_starting_points,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            seed=seed,
            evaluation_budget=evaluation_budget,
            negate=negate,
        )
        x0 = f.x0

        problem = Problem(f, x0)

        return problem


if __name__ == "__main__":
    from poli.core.registry import register_problem

    gfp_problem_factory = GFPCBasProblemFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__protein_cbas",
    )
    # NOTE: default problem is GP problem
    gfp_problem_factory.create(seed=12)
    # instantiate different types of CBas problems:
    gfp_problem_factory_vae = GFPCBasProblemFactory(problem_type="vae")
    register_problem(
        gfp_problem_factory_vae,
        conda_environment_name="poli__protein_cbas",
    )
    gfp_problem_factory_vae.create(seed=12, problem_type="vae")
    gfp_problem_factory_elbo = GFPCBasProblemFactory(problem_type="elbo")
    register_problem(
        gfp_problem_factory_elbo,
        conda_environment_name="poli__protein_cbas",
    )
    gfp_problem_factory_elbo.create(seed=12, problem_type="elbo")
