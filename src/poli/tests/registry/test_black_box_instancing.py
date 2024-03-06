from pathlib import Path
import pytest

import numpy as np

from poli.objective_repository import (
    AlohaBlackBox,
    DockstringBlackBox,
    DRD3BlackBox,
    FoldXRFPLamboBlackBox,
    FoldXSASABlackBox,
    FoldXStabilityBlackBox,
    FoldXStabilityAndSASABlackBox,
    GFPCBasBlackBox,
    GFPSelectionBlackBox,
    PenalizedLogPLamboBlackBox,
    RaspBlackBox,
    LogPBlackBox,
    QEDBlackBox,
    SABlackBox,
    SuperMarioBrosBlackBox,
    ToyContinuousBlackBox,
    WhiteNoiseBlackBox,
)

TESTS_FOLDER = Path(__file__).parent.parent.resolve()

SEED = np.random.randint(0, 1000)

test_data = [
    ("aloha", AlohaBlackBox, {}),
    (
        "dockstring",
        DockstringBlackBox,
        {"target_name": "drd2", "string_representation": "SMILES"},
    ),
    (
        "drd3_docking",
        DRD3BlackBox,
        {"string_representation": "SMILES", "force_isolation": True},
    ),
    (
        "foldx_rfp_lambo",
        FoldXRFPLamboBlackBox,
        {},
    ),
    (
        "foldx_sasa",
        FoldXSASABlackBox,
        {
            "wildtype_pdb_path": TESTS_FOLDER
            / "static_files_for_tests"
            / "101m_Repair.pdb",
            "force_isolation": True,
        },
    ),
    (
        "foldx_stability",
        FoldXStabilityBlackBox,
        {
            "wildtype_pdb_path": TESTS_FOLDER
            / "static_files_for_tests"
            / "101m_Repair.pdb",
            "force_isolation": True,
        },
    ),
    (
        "foldx_stability_and_sasa",
        FoldXStabilityAndSASABlackBox,
        {
            "wildtype_pdb_path": TESTS_FOLDER
            / "static_files_for_tests"
            / "101m_Repair.pdb",
            "force_isolation": True,
            "verbose": True,
        },
    ),
    (
        "gfp_cbas",
        GFPCBasBlackBox,
        {
            "problem_type": "gp",
            "force_isolation": True,
        },
    ),
    (
        "gfp_cbas",
        GFPCBasBlackBox,
        {
            "problem_type": "vae",
            "force_isolation": True,
        },
    ),
    (
        "gfp_cbas",
        GFPCBasBlackBox,
        {
            "problem_type": "elbo",
            "force_isolation": True,
        },
    ),
    (
        "gfp_select",
        GFPSelectionBlackBox,
        {
            "force_isolation": True,
        },
    ),
    (
        "penalized_logp_lambo",
        PenalizedLogPLamboBlackBox,
        {
            "force_isolation": False,
        },
    ),
    (
        "rasp",
        RaspBlackBox,
        {
            "wildtype_pdb_path": TESTS_FOLDER / "static_files_for_tests" / "3ned.pdb",
            "force_isolation": False,
        },
    ),
    (
        "rdkit_logp",
        LogPBlackBox,
        {},
    ),
    (
        "rdkit_qed",
        QEDBlackBox,
        {},
    ),
    (
        "rfp_foldx_stability_and_sasa",
        FoldXStabilityAndSASABlackBox,
        {
            "wildtype_pdb_path": [
                TESTS_FOLDER / "static_files_for_tests" / folder / "wt_input_Repair.pdb"
                for folder in [
                    "2vad_A",
                    "2vae_A",
                    "3e5v_A",
                    "3ned_A",
                    "5lk4_A",
                    "6aa7_A",
                ]
            ],
            "verbose": True,
            "batch_size": 1,
            "parallelize": True,
        },
    ),
    (
        "sa_tdc",
        SABlackBox,
        {
            "force_isolation": True,
        },
    ),
    (
        "super_mario_bros",
        SuperMarioBrosBlackBox,
        {
            "force_isolation": True,
        },
    ),
    (
        "toy_continuous_problem",
        ToyContinuousBlackBox,
        {"function_name": "ackley_function_01"},
    ),
    ("white_noise", WhiteNoiseBlackBox, {}),
]


@pytest.mark.parametrize(
    "black_box_name, black_box_class, kwargs_for_black_box",
    test_data,
)
@pytest.mark.slow()
def test_instancing_a_black_box_both_ways_matches(
    black_box_name, black_box_class, kwargs_for_black_box
):
    from poli import create
    from poli.core.util.seeding import seed_python_numpy_and_torch

    problem = create(
        name=black_box_name,
        seed=SEED,
        **kwargs_for_black_box,
    )
    x0 = problem.x0
    if black_box_name == "foldx_rfp_lambo":
        x0 = x0[0].reshape(1, -1)
    elif black_box_name == "gfp_select":
        x0 = x0[:10]
    y0 = problem.black_box(x0)

    seed_python_numpy_and_torch(SEED)
    f = black_box_class(**kwargs_for_black_box)
    y0_ = f(x0)

    # if problem.info.deterministic:
    # TODO: ask Richard about gfp select and rfp_foldx.
    if black_box_name == "gfp_select":
        return
    elif black_box_name == "rfp_foldx_stability_and_sasa":
        assert np.allclose(y0[:, 0], y0_[:, 0], atol=1.0)
        assert np.allclose(y0_[:, 1], y0[:, 1], atol=2.0)
    else:
        assert np.allclose(y0_, y0)


if __name__ == "__main__":
    for black_box_name, black_box_class, kwargs in test_data:
        test_instancing_a_black_box_both_ways_matches(
            black_box_name, black_box_class, kwargs
        )
