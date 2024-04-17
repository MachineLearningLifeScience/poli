from typing import Literal, Union

from poli.core.problem import Problem
from poli.core.abstract_benchmark import AbstractBenchmark

from poli.objective_repository import (
    AlbuterolSimilarityProblemFactory,
    AmlodipineMPOProblemFactory,
    CelecoxibRediscoveryProblemFactory,
    DecoHopProblemFactory,
    FexofenadineMPOProblemFactory,
    IsomerC7H8N2O2ProblemFactory,
    IsomerC9H10N2O2PF2ClProblemFactory,
    Median1ProblemFactory,
    Median2ProblemFactory,
    MestranolSimilarityProblemFactory,
    OsimetrinibMPOProblemFactory,
    PerindoprilMPOProblemFactory,
    RanolazineMPOProblemFactory,
    LogPProblemFactory,
    QEDProblemFactory,
    SAProblemFactory,
    ScaffoldHopProblemFactory,
    SitagliptinMPOProblemFactory,
    ThiothixeneRediscoveryProblemFactory,
    TroglitazoneRediscoveryProblemFactory,
    ValsartanSMARTSProblemFactory,
    ZaleplonMPOProblemFactory,
)


class GuacamolGoalOrientedBenchmark(AbstractBenchmark):
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
            seed=seed,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        self.problem_factories = [
            AlbuterolSimilarityProblemFactory(),
            AmlodipineMPOProblemFactory(),
            CelecoxibRediscoveryProblemFactory(),
            DecoHopProblemFactory(),
            FexofenadineMPOProblemFactory(),
            IsomerC7H8N2O2ProblemFactory(),
            IsomerC9H10N2O2PF2ClProblemFactory(),
            Median1ProblemFactory(),
            Median2ProblemFactory(),
            MestranolSimilarityProblemFactory(),
            OsimetrinibMPOProblemFactory(),
            PerindoprilMPOProblemFactory(),
            RanolazineMPOProblemFactory(),
            LogPProblemFactory(),
            QEDProblemFactory(),
            SAProblemFactory(),
            ScaffoldHopProblemFactory(),
            SitagliptinMPOProblemFactory(),
            ThiothixeneRediscoveryProblemFactory(),
            TroglitazoneRediscoveryProblemFactory(),
            ValsartanSMARTSProblemFactory(),
            ZaleplonMPOProblemFactory(),
        ]
        self.string_representation = string_representation

    def _initialize_problem(self, index: int) -> Problem:
        problem_factory = self.problem_factories[index]

        problem = problem_factory.create(
            string_representation=self.string_representation,
            seed=self.seed,
            batch_size=self.batch_size,
            parallelize=self.parallelize,
            num_workers=self.num_workers,
            evaluation_budget=self.evaluation_budget,
        )

        return problem
