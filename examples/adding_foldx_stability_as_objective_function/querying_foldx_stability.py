from pathlib import Path

from poli import objective_factory
from poli.core.registry import register_problem

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_path = THIS_DIR / "101m_Repair.pdb"

    problem_info, f, x0, y0, run_info = objective_factory.create(
        "foldx_stability",
        seed=0,
        observer=None,
        wildtype_pdb_path=wildtype_pdb_path,
    )

    print(f"Initial sequence: {x0}")
    print(f"Initial score: {y0}")
