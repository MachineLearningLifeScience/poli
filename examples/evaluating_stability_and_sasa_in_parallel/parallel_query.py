from pathlib import Path

from poli import objective_factory

if __name__ == "__main__":
    wildtype_pdb_path = Path(__file__).parent / "101m_Repair.pdb"

    problem_info, f, x0, y0, _ = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=10 * [wildtype_pdb_path],
        parallelize=False,
        # num_workers=6,
        batch_size=1,
    )
