"""All objective factories and black boxes inside poli.
"""

from pathlib import Path

# These can be imported from the base environment.
from .white_noise.register import WhiteNoiseProblemFactory, WhiteNoiseBlackBox
from .aloha.register import AlohaProblemFactory, AlohaBlackBox
from .toy_continuous_problem.register import (
    ToyContinuousProblemFactory,
    ToyContinuousBlackBox,
)
from .dockstring.register import DockstringProblemFactory, DockstringBlackBox
from .drd3_docking.register import DRD3ProblemFactory, DRD3BlackBox
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
from .penalized_logp_lambo.register import (
    PenalizedLogPLamboBlackBox,
    PenalizedLogPLamboProblemFactory,
)
from .rasp.register import RaspBlackBox, RaspProblemFactory
from .rdkit_logp.register import LogPBlackBox, LogPProblemFactory
from .rdkit_qed.register import QEDBlackBox, QEDProblemFactory
from .rfp_foldx_stability_and_sasa.register import (
    RFPFoldXStabilityAndSASAProblemFactory,
)
from .sa_tdc.register import SAProblemFactory, SABlackBox
from .super_mario_bros.register import (
    SuperMarioBrosProblemFactory,
    SuperMarioBrosBlackBox,
)


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

AVAILABLE_PROBLEM_FACTORIES = {
    "aloha": AlohaProblemFactory,
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
    "rfp_foldx_stability_and_sasa": RFPFoldXStabilityAndSASAProblemFactory,
    "sa_tdc": SAProblemFactory,
    "super_mario_bros": SuperMarioBrosProblemFactory,
    "white_noise": WhiteNoiseProblemFactory,
    "toy_continuous_problem": ToyContinuousProblemFactory,
}

AVAILABLE_BLACK_BOXES = {
    "aloha": AlohaBlackBox,
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
    "rfp_foldx_stability_and_sasa": FoldXStabilityAndSASABlackBox,
    "sa_tdc": SABlackBox,
    "super_mario_bros": SuperMarioBrosBlackBox,
    "white_noise": WhiteNoiseBlackBox,
    "toy_continuous_problem": ToyContinuousBlackBox,
}


def get_problems():
    return list(AVAILABLE_PROBLEM_FACTORIES.keys())
