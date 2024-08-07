from __future__ import annotations

from typing import List, Union

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem import Problem


class AbstractBenchmark:
    problem_factories: List[AbstractProblemFactory]
    index: int = 0

    def __init__(
        self,
        seed: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        parallelize: bool = False,
        num_workers: Union[int, None] = None,
        evaluation_budget: int = float("inf"),
    ) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.parallelize = parallelize
        self.num_workers = num_workers
        self.evaluation_budget = evaluation_budget

    def __len__(self) -> int:
        return len(self.problem_factories)

    def __getitem__(self, index: int) -> Problem:
        return self._initialize_problem(index)

    def __next__(self) -> Problem:
        if self.index < len(self.problem_factories):
            self.index += 1
            return self._initialize_problem(self.index - 1)
        else:
            raise StopIteration

    def _initialize_problem(self, index: int) -> Problem:
        # An abstract method that should be implemented by the subclasses.
        raise NotImplementedError

    @property
    def info(self) -> str:
        raise NotImplementedError

    @property
    def problem_names(self) -> List[str]:
        return [
            problem_factory.__module__.replace(
                "poli.objective_repository.", ""
            ).replace(".register", "")
            for problem_factory in self.problem_factories
        ]
