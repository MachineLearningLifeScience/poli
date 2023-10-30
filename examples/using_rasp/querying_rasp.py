"""
This is an example of how to use the RaSP black box
and objective factory.
"""
from pathlib import Path

from poli.objective_repository.rasp.register import RaspProblemFactory


THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_file = THIS_DIR / "101m.pdb"

    f, x0, y0 = RaspProblemFactory().create(wildtype_pdb_path=wildtype_pdb_file)
    print(x0, y0)
