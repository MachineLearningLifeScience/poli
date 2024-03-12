"""
In this example, we create both a RaSP and a FoldX objective function
and we compare their predictions of stability.

To run this example, you will need to have FoldX installed. Follow our
instructions for set-up here:

https://machinelearninglifescience.github.io/poli-docs/understanding_foldx/00-installing-foldx.html
"""

from pathlib import Path

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_paths_for_foldx = list(
        (THIS_DIR / "example_pdbs").glob("*_Repair.pdb")
    )
    wildtype_pdb_paths_for_rasp = list((THIS_DIR / "example_pdbs").glob("*.pdb"))
    wildtype_pdb_paths_for_rasp = [
        path_
        for path_ in wildtype_pdb_paths_for_rasp
        if "_Repair" not in str(path_.name)
    ]

    foldx_problem = objective_factory.create(
        name="foldx_stability",
        wildtype_pdb_path=wildtype_pdb_paths_for_foldx,
        batch_size=1,
    )
    f_foldx, x0 = foldx_problem.black_box, foldx_problem.x0

    print(f_foldx(x0))

    f_foldx.terminate()

    rasp_problem = objective_factory.create(
        name="rasp",
        wildtype_pdb_path=wildtype_pdb_paths_for_rasp,
    )
    f_rasp, x0 = rasp_problem.black_box, rasp_problem.x0

    print(f_rasp(x0))

    f_rasp.terminate()
