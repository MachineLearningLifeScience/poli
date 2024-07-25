from pathlib import Path

import numpy as np
import pytest

from poli import objective_factory
from poli.objective_repository import (
    AlbuterolSimilarityBlackBox,
    AmlodipineMPOBlackBox,
    CelecoxibRediscoveryBlackBox,
    DecoHopBlackBox,
    DRD2BlackBox,
    FexofenadineMPOBlackBox,
    GSK3BetaBlackBox,
    IsomerC7H8N2O2BlackBox,
    IsomerC9H10N2O2PF2ClBlackBox,
    JNK3BlackBox,
    Median1BlackBox,
    Median2BlackBox,
    MestranolSimilarityBlackBox,
    OsimetrinibMPOBlackBox,
    PerindoprilMPOBlackBox,
    RanolazineMPOBlackBox,
    SABlackBox,
    ScaffoldHopBlackBox,
    SitagliptinMPOBlackBox,
    ThiothixeneRediscoveryBlackBox,
    TroglitazoneRediscoveryBlackBox,
    ValsartanSMARTSBlackBox,
    ZaleplonMPOBlackBox,
)

THIS_DIR = Path(__file__).parent.resolve()

SEED = np.random.randint(0, 1000)


@pytest.mark.poli__lambo
def test_penalized_logp_lambo():
    """
    Testing whether we can register the logp problem
    from lambo.
    """

    _ = pytest.importorskip("lambo")

    # Using create
    _ = objective_factory.create(name="penalized_logp_lambo")


@pytest.mark.poli__dockstring
def test_querying_dockstring_using_smiles():
    """
    In this test, we force-register and query dockstring.
    """

    problem = objective_factory.create(
        name="dockstring",
        target_name="DRD2",
        string_representation="SMILES",
    )
    f = problem.black_box

    # Docking another smiles
    x1 = np.array([list("CC(=O)OC1=CC=CC=C1C(=O)O")])
    _ = f(x1)

    f.terminate()


@pytest.mark.poli__dockstring
def test_querying_dockstring_using_selfies():
    """
    In this test, we check whether dockstring still
    works when using SELFIES instead of SMILES.
    """

    problem = objective_factory.create(
        name="dockstring",
        target_name="ABL1",
        string_representation="SELFIES",
    )
    f = problem.black_box

    # Docking another smiles
    selfies_aspirin = np.array(
        [
            [
                "[C]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[Ring1]",
                "[=Branch1]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
            ]
        ]
    )

    _ = f(selfies_aspirin)
    f.terminate()


test_data_for_pmo = [
    ("gsk3_beta", GSK3BetaBlackBox, {"string_representation": "SMILES"}, 0.03),
    (
        "drd2_docking",
        DRD2BlackBox,
        {"string_representation": "SMILES"},
        0.0015465365340340924,
    ),
    (
        "sa_tdc",
        SABlackBox,
        {"string_representation": "SMILES"},
        2.706977149048555,
    ),
    (
        "celecoxib_rediscovery",
        CelecoxibRediscoveryBlackBox,
        {"string_representation": "SMILES"},
        0.14728682170542637,
    ),
    (
        "thiothixene_rediscovery",
        ThiothixeneRediscoveryBlackBox,
        {"string_representation": "SMILES"},
        0.17391304347826086,
    ),
    (
        "troglitazone_rediscovery",
        TroglitazoneRediscoveryBlackBox,
        {"string_representation": "SMILES"},
        0.24427480916030533,
    ),
    (
        "albuterol_similarity",
        AlbuterolSimilarityBlackBox,
        {"string_representation": "SMILES"},
        0.2772277227722772,
    ),
    (
        "mestranol_similarity",
        MestranolSimilarityBlackBox,
        {"string_representation": "SMILES"},
        0.19460880999342536,
    ),
    (
        "amlodipine_mpo",
        AmlodipineMPOBlackBox,
        {"string_representation": "SMILES"},
        0.461083967620704,
    ),
    (
        "fexofenadine_mpo",
        FexofenadineMPOBlackBox,
        {"string_representation": "SMILES"},
        0.4336446174984538,
    ),
    (
        "osimetrinib_mpo",
        OsimetrinibMPOBlackBox,
        {"string_representation": "SMILES"},
        0.09011742702110873,
    ),
    (
        "perindopril_mpo",
        PerindoprilMPOBlackBox,
        {"string_representation": "SMILES"},
        0.36023741111440966,
    ),
    (
        "ranolazine_mpo",
        RanolazineMPOBlackBox,
        {"string_representation": "SMILES"},
        0.29285467466584664,
    ),
    # The following two have a discrepancy with
    # the TDC docs. An issue has been raised (#244)
    (
        "sitagliptin_mpo",
        SitagliptinMPOBlackBox,
        {"string_representation": "SMILES"},
        3.34970667598234e-12,
    ),
    (
        "zaleplon_mpo",
        ZaleplonMPOBlackBox,
        {"string_representation": "SMILES"},
        0.0019017991803329235,
    ),
    # The following should actually output 0.5338365434669443,
    # but TDC has a discrepancy. An issue has been raised (#244)
    (
        "deco_hop",
        DecoHopBlackBox,
        {"string_representation": "SMILES"},
        0.5338365434669443,
    ),
    (
        "scaffold_hop",
        ScaffoldHopBlackBox,
        {"string_representation": "SMILES"},
        0.38446411012782694,
    ),
    # The following two have a discrepancy with
    # the TDC docs. An issue has been raised (#244)
    (
        "isomer_c7h8n2o2",
        IsomerC7H8N2O2BlackBox,
        {"string_representation": "SMILES"},
        2.1987591132394053e-34,
    ),
    (
        "isomer_c9h10n2o2pf2cl",
        IsomerC9H10N2O2PF2ClBlackBox,
        {"string_representation": "SMILES"},
        1.713908431542013e-15,
    ),
    (
        "median_1",
        Median1BlackBox,
        {"string_representation": "SMILES"},
        0.09722243533981723,
    ),
    (
        "median_2",
        Median2BlackBox,
        {"string_representation": "SMILES"},
        0.12259690287307903,
    ),
    (
        "valsartan_smarts",
        ValsartanSMARTSBlackBox,
        {"string_representation": "SMILES"},
        0.0,
    ),
    ("jnk3", JNK3BlackBox, {"string_representation": "SMILES"}, 0.01),
]


@pytest.mark.poli__tdc
@pytest.mark.parametrize(
    "black_box_name, black_box_class, kwargs_for_black_box, value_to_check",
    test_data_for_pmo,
)
def test_pmo_black_boxes(
    black_box_name, black_box_class, kwargs_for_black_box, value_to_check
):
    from poli import create
    from poli.core.util.seeding import seed_python_numpy_and_torch

    problem = create(
        name=black_box_name,
        seed=SEED,
        **kwargs_for_black_box,
    )
    x0 = problem.x0
    y0 = problem.black_box(x0)

    seed_python_numpy_and_torch(seed=SEED)
    f = black_box_class(**kwargs_for_black_box)
    y0_ = f(x0)

    assert np.allclose(y0_, y0)
    if value_to_check is not None:
        assert (y0_ == value_to_check).all()
