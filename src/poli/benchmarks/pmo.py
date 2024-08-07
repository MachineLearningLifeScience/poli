"""Implementation of the Practical Molecular Optimization benchmark

The Practical Molecular Optimization (PMO) benchmark [1] is a collection of
objective functions for molecular optimization. It is a superset of GuacaMol [2],
including additional tasks such as DRD2, JNK3, and GSK3Beta.

Our implementation heavily relies on the Therapeutics Data Commons [3], which
provides these black boxes.

Check the details in the respective papers.

References
----------
[1] Brown, Nathan, Marco Fiscato, Marwin H.S. Segler, and Alain C. Vaucher.
    “GuacaMol: Benchmarking Models for de Novo Molecular Design.”
    Journal of Chemical Information and Modeling 59, no. 3 (March 25, 2019):
    1096-1108. https://doi.org/10.1021/acs.jcim.8b00839.

[2] Gao, Wenhao, Tianfan Fu, Jimeng Sun, and Connor W. Coley.
    “Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization,” 2022.
    https://openreview.net/forum?id=yCZRdI0Y7G.

[3] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec,
    Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik.
    “Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.”
    Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021.

"""

from typing import Literal, Union

from poli.objective_repository import (
    DRD2ProblemFactory,
    GSK3BetaProblemFactory,
    JNK3ProblemFactory,
)

from .guacamol import GuacaMolGoalDirectedBenchmark


class PMOBenchmark(GuacaMolGoalDirectedBenchmark):
    """Practical Molecular Optimization benchmark

    The Practical Molecular Optimization (PMO) benchmark [1] is a
    collection of objective functions for molecular optimization.
    It is a superset of GuacaMol [2], including additional tasks such
    as DRD2, JNK3, and GSK3Beta.

    Our implementation heavily relies on the Therapeutics Data Commons [3],
    which provides these black boxes.

    Check the details in the respective papers.

    References
    ----------
    [1] Brown, Nathan, Marco Fiscato, Marwin H.S. Segler, and Alain C. Vaucher.
        “GuacaMol: Benchmarking Models for de Novo Molecular Design.”
        Journal of Chemical Information and Modeling 59, no. 3 (March 25, 2019):
        1096-1108. https://doi.org/10.1021/acs.jcim.8b00839.

    [2] Gao, Wenhao, Tianfan Fu, Jimeng Sun, and Connor W. Coley.
        “Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization,” 2022.
        https://openreview.net/forum?id=yCZRdI0Y7G.

    [3] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec,
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
