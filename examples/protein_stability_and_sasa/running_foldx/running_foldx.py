from pathlib import Path

from poli.repository import FoldXStabilityAndSASAProblemFactory

if __name__ == "__main__":
    pdb_file = Path(__file__).parent / "101m_Repair.pdb"
    problem = FoldXStabilityAndSASAProblemFactory().create(
        wildtype_pdb_path=pdb_file,
    )
    f, x0 = problem.black_box, problem.x0

    print(f)
    print(x0)
    print(f(x0))
