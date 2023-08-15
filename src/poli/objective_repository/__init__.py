from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()

# The objective repository is made of
# all the directories that are here:
AVAILABLE_OBJECTIVES = [
    str(d.name)
    for d in THIS_DIR.glob("*")
    if d.is_dir() and d.name != "__pycache__"
]
