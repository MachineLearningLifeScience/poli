"""
Implementation of the GuacaMol benchmark.

GuacaMol [1] is a benchmark for small molecule optimization. It was
developed in the context of generative models for molecular design.
Originally, it had two components: distirbution-learning and goal-directed
benchmarks. This implementation focuses on goal-directed benchmarks.

Our implementation heavily relies on Therapeutics Data Commons [2].

Check the details in the original paper [1].

References
----------
[1] Brown, Nathan, Marco Fiscato, Marwin H.S. Segler, and Alain C. Vaucher.
    “GuacaMol: Benchmarking Models for de Novo Molecular Design.”
    Journal of Chemical Information and Modeling 59, no. 3 (March 25, 2019):
    1096-1108. https://doi.org/10.1021/acs.jcim.8b00839.

[2] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec,
    Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik.
    “Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.”
    Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021.
"""

from typing import Literal, Union

from poli.core.abstract_benchmark import AbstractBenchmark
from poli.core.problem import Problem
from poli.objective_repository import (
    AlbuterolSimilarityProblemFactory,
    AmlodipineMPOProblemFactory,
    CelecoxibRediscoveryProblemFactory,
    DecoHopProblemFactory,
    FexofenadineMPOProblemFactory,
    IsomerC7H8N2O2ProblemFactory,
    IsomerC9H10N2O2PF2ClProblemFactory,
    LogPProblemFactory,
    Median1ProblemFactory,
    Median2ProblemFactory,
    MestranolSimilarityProblemFactory,
    OsimetrinibMPOProblemFactory,
    PerindoprilMPOProblemFactory,
    QEDProblemFactory,
    RanolazineMPOProblemFactory,
    SAProblemFactory,
    ScaffoldHopProblemFactory,
    SitagliptinMPOProblemFactory,
    ThiothixeneRediscoveryProblemFactory,
    TroglitazoneRediscoveryProblemFactory,
    ValsartanSMARTSProblemFactory,
    ZaleplonMPOProblemFactory,
)


class GuacaMolGoalDirectedBenchmark(AbstractBenchmark):
    """GuacaMol benchmark for goal-directed tasks

    GuacaMol [1] is a benchmark for small molecule optimization. It was
    developed in the context of generative models for molecular design.
    Originally, it had two components: distirbution-learning and goal-directed
    benchmarks. This implementation focuses on goal-directed benchmarks.

    Our implementation heavily relies on Therapeutics Data Commons [2].

    Check the details in the original paper [1].

    References
    ----------
    [1] Brown, Nathan, Marco Fiscato, Marwin H.S. Segler, and Alain C. Vaucher.
        “GuacaMol: Benchmarking Models for de Novo Molecular Design.”
        Journal of Chemical Information and Modeling 59, no. 3 (March 25, 2019):
        1096-1108. https://doi.org/10.1021/acs.jcim.8b00839.

    [2] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec,
        Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik.
        “Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.”
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021.
    """

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
