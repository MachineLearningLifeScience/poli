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
except ImportError:
    pass
