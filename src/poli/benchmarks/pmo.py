from typing import Literal, Union

from poli.objective_repository import (
    DRD2ProblemFactory,
    JNK3ProblemFactory,
    GSK3BetaProblemFactory,
)

from .guacamol import GuacamolGoalOrientedBenchmark


class PMOBenchmark(GuacamolGoalOrientedBenchmark):
    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"],
        seed: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        parallelize: bool = False,
        num_workers: Union[int, None] = None,
        evaluation_budget: int = float("inf"),
    ) -> None:
        super().__init__(
            string_representation=string_representation,
            seed=seed,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        self.problem_factories.extend(
            [
                JNK3ProblemFactory(),
                GSK3BetaProblemFactory(),
                DRD2ProblemFactory(),
            ]
        )
