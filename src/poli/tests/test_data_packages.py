import pytest

from poli.core.chemistry.data_packages.random_molecules_data_package import (
    RandomMoleculesDataPackage,
)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seed_determinism_for_random_molecule_data_packages(seed):
    data_package_1 = RandomMoleculesDataPackage(
        string_representation="SMILES", seed=seed
    )

    data_package_2 = RandomMoleculesDataPackage(
        string_representation="SMILES", seed=seed
    )

    assert (data_package_1.unsupervised_data == data_package_2.unsupervised_data).all()
