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


THIS_DIR = Path(__file__).parent.resolve()

# The objective repository is made of
# all the directories that are here:
AVAILABLE_OBJECTIVES = []
for d in THIS_DIR.glob("*"):
    if d.is_dir() and d.name != "__pycache__":
        AVAILABLE_OBJECTIVES.append(d.name)

        if (d / "isolated_black_box.py").exists():
            AVAILABLE_OBJECTIVES.append(f"{d.name}__isolated")

AVAILABLE_OBJECTIVES = sorted(AVAILABLE_OBJECTIVES)

AVAILABLE_PROBLEM_FACTORIES = {
    "white_noise": WhiteNoiseProblemFactory,
    "aloha": AlohaProblemFactory,
    "toy_continuous_problem": ToyContinuousProblemFactory,
}

AVAILABLE_BLACK_BOXES = {
    "white_noise": WhiteNoiseBlackBox,
    "aloha": AlohaBlackBox,
    "toy_continuous_problem": ToyContinuousBlackBox,
}

try:
    # TODO: the case of SMB is a little bit more delicate, since
    # we actually have dependencies beyond Python.
    # TODO: check that JAVA is available in the environment.
    from .super_mario_bros.register import SMBProblemFactory, SMBBlackBox

    AVAILABLE_PROBLEM_FACTORIES["super_mario_bros"] = SMBProblemFactory
    AVAILABLE_BLACK_BOXES["super_mario_bros"] = SMBBlackBox
except ImportError:
    pass

try:
    from .rdkit_qed.register import QEDProblemFactory, QEDBlackBox

    AVAILABLE_PROBLEM_FACTORIES["rdkit_qed"] = QEDProblemFactory
    AVAILABLE_BLACK_BOXES["rdkit_qed"] = QEDBlackBox
except ImportError:
    pass

try:
    from .rdkit_logp.register import LogPProblemFactory, LogPBlackBox

    AVAILABLE_PROBLEM_FACTORIES["rdkit_logp"] = LogPProblemFactory
    AVAILABLE_BLACK_BOXES["rdkit_logp"] = LogPBlackBox
except ImportError:
    pass

try:
    from .foldx_stability.register import (
        FoldXStabilityProblemFactory,
        FoldXStabilityBlackBox,
    )

    AVAILABLE_PROBLEM_FACTORIES["foldx_stability"] = FoldXStabilityProblemFactory
    AVAILABLE_BLACK_BOXES["foldx_stability"] = FoldXStabilityBlackBox
except (ImportError, FileNotFoundError):
    pass

try:
    from .foldx_sasa.register import FoldXSASAProblemFactory, FoldXSASABlackBox

    AVAILABLE_PROBLEM_FACTORIES["foldx_sasa"] = FoldXSASAProblemFactory
    AVAILABLE_BLACK_BOXES["foldx_sasa"] = FoldXSASABlackBox
except (ImportError, FileNotFoundError):
    pass


try:
    from .foldx_rfp_lambo.register import RFPWrapperFactory, RFPWrapper

    AVAILABLE_PROBLEM_FACTORIES["foldx_rfp_lambo"] = RFPWrapperFactory
    AVAILABLE_BLACK_BOXES["foldx_rfp_lambo"] = RFPWrapper
except (ImportError, FileNotFoundError):
    pass


try:
    from .foldx_stability_and_sasa.register import (
        FoldXStabilityAndSASAProblemFactory,
        FoldXStabilityAndSASABlackBox,
    )

    AVAILABLE_PROBLEM_FACTORIES["foldx_stability_and_sasa"] = (
        FoldXStabilityAndSASAProblemFactory
    )
    AVAILABLE_BLACK_BOXES["foldx_stability_and_sasa"] = FoldXStabilityAndSASABlackBox
except (ImportError, FileNotFoundError):
    pass


try:
    from .rfp_foldx_stability_and_sasa.register import (
        RFPFoldXStabilityAndSASAProblemFactory,
        RFPFoldXStabilityAndSASABlackBox,
    )

    AVAILABLE_PROBLEM_FACTORIES["rfp_foldx_stability_and_sasa"] = (
        RFPFoldXStabilityAndSASAProblemFactory
    )
    AVAILABLE_BLACK_BOXES["rfp_foldx_stability_and_sasa"] = (
        RFPFoldXStabilityAndSASABlackBox
    )
except (ImportError, FileNotFoundError):
    pass


try:
    from .penalized_logp_lambo.register import (
        PenalizedLogPLamboProblemFactory,
        PenalizedLogPLamboBlackBox,
    )

    AVAILABLE_PROBLEM_FACTORIES["penalized_logp_lambo"] = (
        PenalizedLogPLamboProblemFactory
    )
    AVAILABLE_BLACK_BOXES["penalized_logp_lambo"] = PenalizedLogPLamboBlackBox
except (ImportError, FileNotFoundError):
    pass

try:
    from .drd3_docking.register import DDR3ProblemFactory, DRD3BlackBox

    AVAILABLE_PROBLEM_FACTORIES["drd3_docking"] = DDR3ProblemFactory
    AVAILABLE_BLACK_BOXES["drd3_docking"] = DRD3BlackBox
except (ImportError, FileNotFoundError):
    pass


try:
    from .gfp_select.register import GFPSelectionProblemFactory, GFPBlackBox

    AVAILABLE_PROBLEM_FACTORIES["gfp_select"] = GFPSelectionProblemFactory
    AVAILABLE_BLACK_BOXES["gfp_select"] = GFPBlackBox
except (ImportError, FileNotFoundError):
    pass


try:
    from .rasp.register import RaspProblemFactory, RaspBlackBox

    AVAILABLE_PROBLEM_FACTORIES["rasp"] = RaspProblemFactory
    AVAILABLE_BLACK_BOXES["rasp"] = RaspBlackBox
except (ImportError, FileNotFoundError):
    pass

try:
    from .dockstring.register import DockstringProblemFactory, DockstringBlackBox

    AVAILABLE_PROBLEM_FACTORIES["dockstring"] = DockstringProblemFactory
    AVAILABLE_BLACK_BOXES["dockstring"] = DockstringBlackBox
except ImportError:
    pass
