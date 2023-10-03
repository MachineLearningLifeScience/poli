from pathlib import Path

# These can be imported from the base environment.
from .white_noise.register import WhiteNoiseProblemFactory
from .aloha.register import AlohaProblemFactory

# These have more complex dependencies
# from .super_mario_bros.register import SuperMarioBrosBlackBox
# from .rdkit_qed.register import QEDBlackBox


THIS_DIR = Path(__file__).parent.resolve()

# The objective repository is made of
# all the directories that are here:
AVAILABLE_OBJECTIVES = sorted(
    [str(d.name) for d in THIS_DIR.glob("*") if d.is_dir() and d.name != "__pycache__"]
)

AVAILABLE_PROBLEM_FACTORIES = {
    "white_noise": WhiteNoiseProblemFactory,
    "aloha": AlohaProblemFactory,
}


try:
    # TODO: the case of SMB is a little bit more delicate, since
    # we actually have dependencies beyond Python.
    from .super_mario_bros.register import SMBProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["super_mario_bros"] = SMBProblemFactory
except ImportError:
    pass

try:
    from .rdkit_qed.register import QEDProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["rdkit_qed"] = QEDProblemFactory
except ImportError:
    pass

try:
    from .rdkit_logp.register import LogPProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["rdkit_logp"] = LogPProblemFactory
except ImportError:
    pass

try:
    from .foldx_stability.register import FoldXStabilityProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["foldx_stability"] = FoldXStabilityProblemFactory
except (ImportError, FileNotFoundError):
    pass

try:
    from .foldx_sasa.register import FoldXSASAProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["foldx_sasa"] = FoldXSASAProblemFactory
except (ImportError, FileNotFoundError):
    pass


try:
    from .foldx_rfp_lambo.register import RFPWrapperFactory

    AVAILABLE_PROBLEM_FACTORIES["foldx_rfp_lambo"] = RFPWrapperFactory
except (ImportError, FileNotFoundError):
    pass


try:
    from .foldx_stability_and_sasa.register import FoldXStabilityAndSASAProblemFactory

    AVAILABLE_PROBLEM_FACTORIES[
        "foldx_stability_and_sasa"
    ] = FoldXStabilityAndSASAProblemFactory
except (ImportError, FileNotFoundError):
    pass


try:
    from .penalized_logp_lambo.register import PenalizedLogPLamboProblemFactory

    AVAILABLE_PROBLEM_FACTORIES[
        "penalized_logp_lambo"
    ] = PenalizedLogPLamboProblemFactory
except (ImportError, FileNotFoundError):
    pass

try:
    from .drd3_docking.register import DDR3ProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["drd3_docking"] = DDR3ProblemFactory
except (ImportError, FileNotFoundError):
    pass


try:
    from .gfp_select.register import GFPSelectionProblemFactory

    AVAILABLE_PROBLEM_FACTORIES["gfp_select"] = GFPSelectionProblemFactory
except (ImportError, FileNotFoundError):
    pass
