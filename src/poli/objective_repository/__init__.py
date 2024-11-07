"""All objective factories and black boxes inside poli.
"""

from pathlib import Path

from .albuterol_similarity.register import (
    AlbuterolSimilarityBlackBox,
    AlbuterolSimilarityProblemFactory,
)
from .aloha.register import AlohaBlackBox, AlohaProblemFactory
from .amlodipine_mpo.register import AmlodipineMPOBlackBox, AmlodipineMPOProblemFactory
from .celecoxib_rediscovery.register import (
    CelecoxibRediscoveryBlackBox,
    CelecoxibRediscoveryProblemFactory,
)
from .deco_hop.register import DecoHopBlackBox, DecoHopProblemFactory
from .dockstring.register import DockstringBlackBox, DockstringProblemFactory
from .drd2_docking.register import DRD2BlackBox, DRD2ProblemFactory
from .drd3_docking.register import DRD3BlackBox, DRD3ProblemFactory
from .ehrlich.register import EhrlichBlackBox, EhrlichProblemFactory
from .ehrlich_holo.register import EhrlichHoloBlackBox, EhrlichHoloProblemFactory
from .fexofenadine_mpo.register import (
    FexofenadineMPOBlackBox,
    FexofenadineMPOProblemFactory,
)
from .foldx_rfp_lambo.register import FoldXRFPLamboBlackBox, FoldXRFPLamboProblemFactory
from .foldx_sasa.register import FoldXSASABlackBox, FoldXSASAProblemFactory
from .foldx_stability.register import (
    FoldXStabilityBlackBox,
    FoldXStabilityProblemFactory,
)
from .foldx_stability_and_sasa.register import (
    FoldXStabilityAndSASABlackBox,
    FoldXStabilityAndSASAProblemFactory,
)
from .gfp_cbas.register import GFPCBasBlackBox, GFPCBasProblemFactory
from .gfp_select.register import GFPSelectionBlackBox, GFPSelectionProblemFactory
from .gsk3_beta.register import GSK3BetaBlackBox, GSK3BetaProblemFactory
from .isomer_c7h8n2o2.register import (
    IsomerC7H8N2O2BlackBox,
    IsomerC7H8N2O2ProblemFactory,
)
from .isomer_c9h10n2o2pf2cl.register import (
    IsomerC9H10N2O2PF2ClBlackBox,
    IsomerC9H10N2O2PF2ClProblemFactory,
)
from .jnk3.register import JNK3BlackBox, JNK3ProblemFactory
from .median_1.register import Median1BlackBox, Median1ProblemFactory
from .median_2.register import Median2BlackBox, Median2ProblemFactory
from .mestranol_similarity.register import (
    MestranolSimilarityBlackBox,
    MestranolSimilarityProblemFactory,
)
from .osimetrinib_mpo.register import (
    OsimetrinibMPOBlackBox,
    OsimetrinibMPOProblemFactory,
)
from .penalized_logp_lambo.register import (
    PenalizedLogPLamboBlackBox,
    PenalizedLogPLamboProblemFactory,
)
from .perindopril_mpo.register import (
    PerindoprilMPOBlackBox,
    PerindoprilMPOProblemFactory,
)
from .ranolazine_mpo.register import RanolazineMPOBlackBox, RanolazineMPOProblemFactory
from .rasp.register import RaspBlackBox, RaspProblemFactory
from .rdkit_logp.register import LogPBlackBox, LogPProblemFactory
from .rdkit_qed.register import QEDBlackBox, QEDProblemFactory
from .rfp_foldx_stability.register import (
    RFPFoldXStabilityBlackBox,
    RFPFoldXStabilityProblemFactory,
)
from .rfp_foldx_stability_and_sasa.register import (
    RFPFoldXStabilityAndSASAProblemFactory,
)
from .rfp_rasp.register import RFPRaspBlackBox, RFPRaspProblemFactory
from .rmf_landscape.register import RMFBlackBox, RMFProblemFactory
from .rosetta_energy.register import RosettaEnergyBlackBox, RosettaEnergyProblemFactory
from .sa_tdc.register import SABlackBox, SAProblemFactory
from .scaffold_hop.register import ScaffoldHopBlackBox, ScaffoldHopProblemFactory
from .sitagliptin_mpo.register import (
    SitagliptinMPOBlackBox,
    SitagliptinMPOProblemFactory,
)
from .super_mario_bros.register import (
    SuperMarioBrosBlackBox,
    SuperMarioBrosProblemFactory,
)
from .thiothixene_rediscovery.register import (
    ThiothixeneRediscoveryBlackBox,
    ThiothixeneRediscoveryProblemFactory,
)
from .toy_continuous_problem.register import (
    ToyContinuousBlackBox,
    ToyContinuousProblemFactory,
)
from .troglitazone_rediscovery.register import (
    TroglitazoneRediscoveryBlackBox,
    TroglitazoneRediscoveryProblemFactory,
)
from .valsartan_smarts.register import (
    ValsartanSMARTSBlackBox,
    ValsartanSMARTSProblemFactory,
)
from .white_noise.register import WhiteNoiseBlackBox, WhiteNoiseProblemFactory
from .zaleplon_mpo.register import ZaleplonMPOBlackBox, ZaleplonMPOProblemFactory

THIS_DIR = Path(__file__).parent.resolve()

# The objective repository is made of
# all the directories that are here:
AVAILABLE_OBJECTIVES = []
for d in THIS_DIR.glob("*"):
    if d.is_dir() and d.name != "__pycache__":
        AVAILABLE_OBJECTIVES.append(d.name)

        if (d / "isolated_function.py").exists():
            AVAILABLE_OBJECTIVES.append(f"{d.name}__isolated")

AVAILABLE_OBJECTIVES = sorted(AVAILABLE_OBJECTIVES)

# Available problem factories
# maps names of problems to their respective
# problem factories
AVAILABLE_PROBLEM_FACTORIES = {
    "aloha": AlohaProblemFactory,
    "ehrlich": EhrlichProblemFactory,
    "ehrlich_holo": EhrlichHoloProblemFactory,
    "dockstring": DockstringProblemFactory,
    "drd3_docking": DRD3ProblemFactory,
    "foldx_rfp_lambo": FoldXRFPLamboProblemFactory,
    "foldx_sasa": FoldXSASAProblemFactory,
    "foldx_stability": FoldXStabilityProblemFactory,
    "foldx_stability_and_sasa": FoldXStabilityAndSASAProblemFactory,
    "gfp_cbas": GFPCBasProblemFactory,
    "gfp_select": GFPSelectionProblemFactory,
    "penalized_logp_lambo": PenalizedLogPLamboProblemFactory,
    "rasp": RaspProblemFactory,
    "rdkit_logp": LogPProblemFactory,
    "rdkit_qed": QEDProblemFactory,
    "rfp_rasp": RFPRaspProblemFactory,
    "rfp_foldx_stability": RFPFoldXStabilityProblemFactory,
    "rfp_foldx_stability_and_sasa": RFPFoldXStabilityAndSASAProblemFactory,
    "rmf_landscape": RMFProblemFactory,
    "rosetta_energy": RosettaEnergyProblemFactory,
    "sa_tdc": SAProblemFactory,
    "super_mario_bros": SuperMarioBrosProblemFactory,
    "white_noise": WhiteNoiseProblemFactory,
    "toy_continuous_problem": ToyContinuousProblemFactory,
    "gsk3_beta": GSK3BetaProblemFactory,
    "drd2_docking": DRD2ProblemFactory,
    "jnk3": JNK3ProblemFactory,
    "celecoxib_rediscovery": CelecoxibRediscoveryProblemFactory,
    "thiothixene_rediscovery": ThiothixeneRediscoveryProblemFactory,
    "troglitazone_rediscovery": TroglitazoneRediscoveryProblemFactory,
    "albuterol_similarity": AlbuterolSimilarityProblemFactory,
    "mestranol_similarity": MestranolSimilarityProblemFactory,
    "amlodipine_mpo": AmlodipineMPOProblemFactory,
    "fexofenadine_mpo": FexofenadineMPOProblemFactory,
    "osimetrinib_mpo": OsimetrinibMPOProblemFactory,
    "perindopril_mpo": PerindoprilMPOProblemFactory,
    "ranolazine_mpo": RanolazineMPOProblemFactory,
    "sitagliptin_mpo": SitagliptinMPOProblemFactory,
    "zaleplon_mpo": ZaleplonMPOProblemFactory,
    "deco_hop": DecoHopProblemFactory,
    "scaffold_hop": ScaffoldHopProblemFactory,
    "isomer_c7h8n2o2": IsomerC7H8N2O2ProblemFactory,
    "isomer_c9h10n2o2pf2cl": IsomerC9H10N2O2PF2ClProblemFactory,
    "median_1": Median1ProblemFactory,
    "median_2": Median2ProblemFactory,
    "valsartan_smarts": ValsartanSMARTSProblemFactory,
}

AVAILABLE_BLACK_BOXES = {
    "aloha": AlohaBlackBox,
    "ehrlich": EhrlichBlackBox,
    "ehrlich_holo": EhrlichHoloBlackBox,
    "dockstring": DockstringBlackBox,
    "drd3_docking": DRD3BlackBox,
    "foldx_rfp_lambo": FoldXRFPLamboBlackBox,
    "foldx_sasa": FoldXSASABlackBox,
    "foldx_stability": FoldXStabilityBlackBox,
    "foldx_stability_and_sasa": FoldXStabilityAndSASABlackBox,
    "gfp_cbas": GFPCBasBlackBox,
    "gfp_select": GFPSelectionBlackBox,
    "penalized_logp_lambo": PenalizedLogPLamboBlackBox,
    "rasp": RaspBlackBox,
    "rdkit_logp": LogPBlackBox,
    "rdkit_qed": QEDBlackBox,
    "rfp_rasp": RFPRaspBlackBox,
    "rfp_foldx_stability": RFPFoldXStabilityBlackBox,
    "rfp_foldx_stability_and_sasa": FoldXStabilityAndSASABlackBox,
    "rmf_landscape": RMFBlackBox,
    "rosetta_energy": RosettaEnergyBlackBox,
    "sa_tdc": SABlackBox,
    "super_mario_bros": SuperMarioBrosBlackBox,
    "white_noise": WhiteNoiseBlackBox,
    "toy_continuous_problem": ToyContinuousBlackBox,
    "gsk3_beta": GSK3BetaBlackBox,
    "drd2_docking": DRD2BlackBox,
    "jnk3": JNK3BlackBox,
    "celecoxib_rediscovery": CelecoxibRediscoveryBlackBox,
    "thiothixene_rediscovery": ThiothixeneRediscoveryBlackBox,
    "troglitazone_rediscovery": TroglitazoneRediscoveryBlackBox,
    "albuterol_similarity": AlbuterolSimilarityBlackBox,
    "mestranol_similarity": MestranolSimilarityBlackBox,
    "amlodipine_mpo": AmlodipineMPOBlackBox,
    "fexofenadine_mpo": FexofenadineMPOBlackBox,
    "osimetrinib_mpo": OsimetrinibMPOBlackBox,
    "perindopril_mpo": PerindoprilMPOBlackBox,
    "ranolazine_mpo": RanolazineMPOBlackBox,
    "sitagliptin_mpo": SitagliptinMPOBlackBox,
    "zaleplon_mpo": ZaleplonMPOBlackBox,
    "deco_hop": DecoHopBlackBox,
    "scaffold_hop": ScaffoldHopBlackBox,
    "isomer_c7h8n2o2": IsomerC7H8N2O2BlackBox,
    "isomer_c9h10n2o2pf2cl": IsomerC9H10N2O2PF2ClBlackBox,
    "median_1": Median1BlackBox,
    "median_2": Median2BlackBox,
    "valsartan_smarts": ValsartanSMARTSBlackBox,
}


def get_problems():
    return sorted(list(AVAILABLE_PROBLEM_FACTORIES.keys()))
