import time
from pathlib import Path

from poli import objective_factory

if __name__ == "__main__":
    wildtype_pdb_path = Path(__file__).parent / "101m_Repair.pdb"

    foldx_problem_in_parallel = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=10 * [wildtype_pdb_path],
        parallelize=True,
        num_workers=5,
        batch_size=10,
        force_register=True,
    )
    f, x0 = foldx_problem_in_parallel.black_box, foldx_problem_in_parallel.x0
    print("Running in parallel")
    stopwatch = time.time()
    print(f(x0))
    stopwatch_parallel = time.time() - stopwatch

    print("Running in serial")
    foldx_problem_in_serial = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=10 * [wildtype_pdb_path],
        parallelize=False,
        # num_workers=5,
        batch_size=1,  # foldx can't run more than 1, unless in parallel.
    )
    f, x0 = foldx_problem_in_serial.black_box, foldx_problem_in_serial.x0
    stopwatch = time.time()
    print(f(x0))
    stopwatch_serial = time.time() - stopwatch

    print("Time it took to run in parallel: ", stopwatch_parallel)
    print("Time it took to run in serial: ", stopwatch_serial)
