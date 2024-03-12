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
    # TODO: to test this, I'd need access to lambo's assets.
    # (
    #     "rfp_foldx_stability_and_sasa",
    #     FoldXStabilityAndSASABlackBox,
    #     {},
    # ),
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
    # TODO: ask Richard about gfp select.
    if black_box_name != "gfp_select":
        assert np.allclose(y0_, y0)


if __name__ == "__main__":
    for black_box_name, black_box_class, kwargs in test_data:
        test_instancing_a_black_box_both_ways_matches(
            black_box_name, black_box_class, kwargs
        )
