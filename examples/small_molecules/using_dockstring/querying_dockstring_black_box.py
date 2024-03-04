"""
In this small example, we create a black box
function for dockstring with the drd2 target.
"""

from poli import objective_factory

if __name__ == "__main__":
    f_dockstring, x0, y0 = objective_factory.create(
        name="dockstring",
        target_name="DRD2",
        string_representation="SMILES",
    )

    print(f"Score of Risperidone: {f_dockstring(x0)}")
