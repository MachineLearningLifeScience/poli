"""
In this small example, we create a black box
function for dockstring with the drd2 target.
"""

from poli import objective_factory

if __name__ == "__main__":
    problem = objective_factory.create(
        name="dockstring",
        target_name="DRD2",
        string_representation="SMILES",
    )
    f_dockstring, x0 = problem.black_box, problem.x0

    print(f"Score of Risperidone: {f_dockstring(x0)}")
