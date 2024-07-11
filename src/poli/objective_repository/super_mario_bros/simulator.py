"""
This script will send things to simulator.jar.

When ran, it lets a human play a level.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from level_utils import clean_level

filepath = Path(__file__).parent.resolve()
JARFILE_PATH = f"{filepath}/simulator.jar"


def test_level_from_int_array(
    level: np.ndarray,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = clean_level(level)
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def test_level_from_str_array(
    level: np.ndarray,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def run_level(
    level: str,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
) -> dict:
    # Run the simulator.jar file with the given level
    if human_player:
        java = subprocess.Popen(
            ["java", "-cp", JARFILE_PATH, "geometry.PlayLevel", level],
            stdout=subprocess.PIPE,
        )
    else:
        java = subprocess.Popen(
            [
                "java",
                "-cp",
                JARFILE_PATH,
                "geometry.EvalLevel",
                level,
                str(max_time),
                str(visualize).lower(),
            ],
            stdout=subprocess.PIPE,
        )

    lines = java.stdout.readlines()
    res = lines[-1]
    res = json.loads(res.decode("utf8"))
    res["level"] = level

    return res
