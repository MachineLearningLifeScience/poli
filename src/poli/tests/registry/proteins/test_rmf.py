import numpy as np
import pytest

from poli import objective_factory
from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES


def test_rmf_is_available():
    """
    Test if rmf_landscape is available when scipy is installed.
    """
    _ = pytest.importorskip("scipy")
    assert "rmf_landscape" in AVAILABLE_PROBLEM_FACTORIES


def test_force_isolation_rmf_landscape():
    """
    Test if we can force-register the rmf_landscape problem.
    """
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype="HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWNPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
        force_isolation=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    assert np.isclose(y0, 0.)
    f.terminate()


def test_rmf_landscape_init():
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype="HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWNPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    assert np.isclose(y0, 0.)
    f.terminate()


def test_rmf_landscape_batch_eval():
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype="HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWNPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
    )
    N = 10
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    x_t = []
    for i in range(N):
        _x = x0.copy()
        _x[i] = np.random.randint(0,20)
        x_t.append(_x)
    x_t = np.vstack(x_t)
    assert x_t.shape[0] == N
    yt = f(x_t)
    assert yt.shape[0] == N
    # TODO evaluate single variant value range
    f.terminate()



if __name__ == "__main__":
    # test_rmf_is_available()
    test_force_isolation_rmf_landscape()