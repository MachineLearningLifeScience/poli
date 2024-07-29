import numpy as np
import pytest

from poli import objective_factory

ref_aa_seq = "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWNPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"


@pytest.mark.poli__rmf
def test_force_isolation_rmf_landscape():
    """
    Test if we can force-register the rmf_landscape problem.
    """
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype=ref_aa_seq,
        kappa=-100,  # keep noise low
        force_isolation=True,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    assert np.isclose(np.round(y0), 0.0)
    f.terminate()


@pytest.mark.poli__rmf
def test_rmf_landscape_init():
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype=ref_aa_seq,
        kappa=-100,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    assert np.isclose(np.round(y0), 0.0)
    f.terminate()


@pytest.mark.poli__rmf
def test_rmf_landscape_batch_eval():
    problem = objective_factory.create(
        name="rmf_landscape",
        wildtype=ref_aa_seq,
    )
    N = 10
    f, x0 = problem.black_box, problem.x0
    _ = f(x0)
    x_t = []
    seq_b = x0.copy()
    seq_b[0, 1] = "Y"
    x_t = np.vstack([seq_b for _ in range(N)])
    assert x_t.shape[0] == N
    yt = f(x_t)
    assert yt.shape[0] == N
    f.terminate()


@pytest.mark.poli__rmf
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rmf_seed_consistent(seed: int):
    mutation_seq = list(ref_aa_seq)
    mutation_seq[int(len(mutation_seq) / 2)] = "A"
    mutation_seq[int(len(mutation_seq) / 4)] = "H"
    mutation_seq = np.array(mutation_seq)[None, :]
    problem_a = objective_factory.create(
        name="rmf_landscape",
        wildtype=ref_aa_seq,
        seed=seed,
    )
    problem_b = objective_factory.create(
        name="rmf_landscape", wildtype=ref_aa_seq, seed=seed
    )
    f_a, x0_a = problem_a.black_box, problem_a.x0
    y0_a = f_a(x0_a)
    f_b, x0_b = problem_b.black_box, problem_b.x0
    y0_b = f_b(x0_b)

    y1_a = f_a(mutation_seq)
    y1_b = f_b(mutation_seq)
    # test equalities
    assert all([x_a == x_b for x_a, x_b in zip(x0_a[0], x0_b[0])])
    assert y0_a == y0_b
    assert y1_a == y1_b  # value for mutated sequences equal
    f_a.terminate()
    f_b.terminate()


@pytest.mark.poli__rmf
@pytest.mark.parametrize("n_mutations", [1, 2, 3])
def test_rmf_num_mutations_expected_val(n_mutations: int):
    from scipy.stats import genpareto

    SEED = 1
    mutation_seq = list(ref_aa_seq)
    for m in range(n_mutations):
        mutation_seq[int(len(mutation_seq) / 2) - m] = "Y"
    mutation_seq = np.array(mutation_seq)[None, :]
    problem = objective_factory.create(
        name="rmf_landscape",
        kappa=-100,  # set kappa <0 for sampling values close to zero
        c=1,  # set constant to one s.t. number mutations negative additive
        wildtype=ref_aa_seq,
        seed=SEED,
    )

    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)
    y1 = f(mutation_seq)

    rnd_state = np.random.default_rng(SEED)
    ref_noise_0 = genpareto.rvs(f.kappa, size=1, random_state=rnd_state)
    ref_noise_1 = genpareto.rvs(f.kappa, size=1, random_state=rnd_state)

    # black-box value minus noisy component should be approximately mutational distance if c==1
    assert np.isclose(np.round(y0 - ref_noise_0), 0)
    assert np.isclose(np.round(y1 - ref_noise_1), -n_mutations)
    f.terminate()
