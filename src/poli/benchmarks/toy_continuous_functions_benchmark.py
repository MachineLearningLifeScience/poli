from typing import Any, Union, List
from poli.core.abstract_benchmark import AbstractBenchmark
from poli.core.problem import Problem

from poli.objective_repository import ToyContinuousProblemFactory
from poli.objective_repository.toy_continuous_problem.toy_continuous_problem import (
    POSSIBLE_FUNCTIONS,
    TWO_DIMENSIONAL_PROBLEMS,
    SIX_DIMENSIONAL_PROBLEMS,
)


class ToyContinuousFunctionsBenchmark(AbstractBenchmark):
    def __init__(
        self,
        n_dimensions: int = 2,
        embed_in: Union[int, None] = None,
        dimensions_to_embed_in: Union[List[int], None] = None,
        seed: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        parallelize: bool = False,
        num_workers: Union[int, None] = None,
        evaluation_budget: Union[int, List[int]] = float("inf"),
    ) -> None:
        super().__init__(
            seed=seed,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.n_dimensions = n_dimensions
        self.embed_in = embed_in
        self.dimensions_to_embed_in = dimensions_to_embed_in
        self.function_names = list(
            (
                set(POSSIBLE_FUNCTIONS)
                - set(TWO_DIMENSIONAL_PROBLEMS)
                - set(SIX_DIMENSIONAL_PROBLEMS)
            )
        )
        self.problem_factories = [ToyContinuousProblemFactory()] * len(
            self.function_names
        )

    def _initialize_problem(self, index: int) -> Problem:
        problem_factory: ToyContinuousProblemFactory = self.problem_factories[index]

        problem = problem_factory.create(
            function_name=self.function_names[index],
            n_dimensions=self.n_dimensions,
            embed_in=self.embed_in,
            dimensions_to_embed_in=self.dimensions_to_embed_in,
            seed=self.seed,
            batch_size=self.batch_size,
            parallelize=self.parallelize,
            num_workers=self.num_workers,
            evaluation_budget=self.evaluation_budget,
        )

        return problem


class EmbeddedBranin2D(AbstractBenchmark):
    def __init__(
        self,
        seed: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        parallelize: bool = False,
        num_workers: Union[int, None] = None,
        evaluation_budget: Union[int, List[int]] = float("inf"),
    ) -> None:
        super().__init__(
            seed,
            batch_size,
            parallelize,
            num_workers,
            evaluation_budget,
        )
        self.embed_in = [5, 10, 25, 50, 100]
        self.problem_factories = [ToyContinuousProblemFactory()] * len(self.embed_in)

    def _initialize_problem(self, index: int) -> Problem:
        problem_factory: ToyContinuousProblemFactory = self.problem_factories[index]

        problem = problem_factory.create(
            function_name="branin_2d",
            embed_in=self.embed_in[index],
            seed=self.seed,
            batch_size=self.batch_size,
            parallelize=self.parallelize,
            num_workers=self.num_workers,
            evaluation_budget=self.evaluation_budget,
        )

        return problem


class EmbeddedHartmann6D(AbstractBenchmark):
    def __init__(
        self,
        seed: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        parallelize: bool = False,
        num_workers: Union[int, None] = None,
        evaluation_budget: Union[int, List[int]] = float("inf"),
    ) -> None:
        super().__init__(
            seed,
            batch_size,
            parallelize,
            num_workers,
            evaluation_budget,
        )
        self.embed_in = [None, 10, 25, 50, 100]
        self.problem_factories = [ToyContinuousProblemFactory()] * len(self.embed_in)

    def _initialize_problem(self, index: int) -> Problem:
        problem_factory: ToyContinuousProblemFactory = self.problem_factories[index]

        if index == 0:
            problem = problem_factory.create(
                function_name="hartmann_6d",
                n_dimensions=6,
                # embed_in=self.embed_in[index],
                seed=self.seed,
                batch_size=self.batch_size,
                parallelize=self.parallelize,
                num_workers=self.num_workers,
                evaluation_budget=self.evaluation_budget,
            )
        else:
            problem = problem_factory.create(
                function_name="hartmann_6d",
                embed_in=self.embed_in[index],
                seed=self.seed,
                batch_size=self.batch_size,
                parallelize=self.parallelize,
                num_workers=self.num_workers,
                evaluation_budget=self.evaluation_budget,
            )

        return problem
