"""
In this example, we show how to use `poli.benchmarks` using the Practical Molecular Optimization [1, 2] benchmark.

In poli, a benchmark is a collection of problems.
Each problem provides a black box and an initial guess.

References
----------

[1] Gao, Wenhao, Tianfan Fu, Jimeng Sun, and Connor W. Coley.
    “Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization,” 2022.
    https://openreview.net/forum?id=yCZRdI0Y7G.

[2] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. “Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.” Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021.

"""

from poli.benchmarks import PMOBenchmark

# Create the benchmark
benchmark = PMOBenchmark(
    string_representation="SELFIES",  # Could also be "SMILES"
)

# Now you can iterate over the problems in the benchmark
for problem in benchmark:
    f, x0 = problem.black_box, problem.x0
    print(problem.info.name)

    # You'd optimize f here...
