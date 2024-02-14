import time
from pathlib import Path

from poli import objective_factory

if __name__ == "__main__":
    wildtype_pdb_path = Path(__file__).parent / "101m_Repair.pdb"

    print("Running in parallel")
    stopwatch = time.time()
    f, x0, y0 = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=10 * [wildtype_pdb_path],
        parallelize=True,
        num_workers=5,
        batch_size=10,
        force_register=True,
    )
    stopwatch_parallel = time.time() - stopwatch

    stopwatch = time.time()
    print("Running in serial")
    f, x0, y0 = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=10 * [wildtype_pdb_path],
        parallelize=False,
        # num_workers=5,
        batch_size=1,  # foldx can't run more than 1, unless in parallel.
    )
    stopwatch_serial = time.time() - stopwatch

    print("Time it took to run in parallel: ", stopwatch_parallel)
    print("Time it took to run in serial: ", stopwatch_serial)
