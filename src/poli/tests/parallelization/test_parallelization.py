from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory

NUM_WORKERS = min(cpu_count(), 2)


def test_parallelization_in_aloha():
    problem = objective_factory.create(
        "aloha", parallelize=True, num_workers=NUM_WORKERS
    )
    f = problem.black_box

    x1 = np.array(
        [
            ["A", "L", "O", "H", "A"],
            ["M", "I", "G", "U", "E"],
            ["F", "L", "E", "A", "S"],
            ["P", "L", "E", "A", "S"],
        ]
    )

    y1 = f(x1)
    assert (y1 == np.array([[5], [0], [1], [1]])).all()


def test_parallelization_in_qed():
    problem = objective_factory.create(
        "rdkit_qed",
        parallelize=True,
        num_workers=NUM_WORKERS,
        batch_size=4,
        string_representation="SMILES",
    )
    f = problem.black_box

    x1 = np.array(
        [
            ["C", "", "", "", ""],
            ["C", "C", "", "", ""],
            ["C", "C", "C", "", ""],
        ]
    )

    y1 = f(x1)
    assert np.isclose(y1, np.array([[0.35978494], [0.37278556], [0.38547066]])).all()


@pytest.mark.poli__protein
def test_parallelization_in_foldx_stability_and_sasa():
    HOME_DIR = Path().home().resolve()
    PATH_TO_FOLDX_FILES = HOME_DIR / "foldx"
    if not PATH_TO_FOLDX_FILES.exists():
        pytest.skip("FoldX is not installed. ")

    if not (PATH_TO_FOLDX_FILES / "foldx").exists():
        pytest.skip("FoldX is not compiled. ")

    wildtype_pdb_path = Path(__file__).parent / "101m_Repair.pdb"
    problem = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=3 * [wildtype_pdb_path],
        parallelize=True,
        num_workers=NUM_WORKERS,
        batch_size=3,
    )

    f, x0 = problem.black_box, problem.x0
    f(x0)
