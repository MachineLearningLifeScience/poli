from pathlib import Path

# These can be imported from the base environment.
from .white_noise.register import WhiteNoiseBlackBox
from .aloha.register import AlohaBlackBox

# These have more complex dependencies
# from .super_mario_bros.register import SuperMarioBrosBlackBox
# from .rdkit_qed.register import QEDBlackBox


THIS_DIR = Path(__file__).parent.resolve()

# The objective repository is made of
# all the directories that are here:
AVAILABLE_OBJECTIVES = sorted(
    [str(d.name) for d in THIS_DIR.glob("*") if d.is_dir() and d.name != "__pycache__"]
)

AVAILABLE_BLACK_BOXES = [
    WhiteNoiseBlackBox,
    AlohaBlackBox,
]
