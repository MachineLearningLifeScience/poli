"""
This script will send things to simulator.jar.

When ran, it lets a human play a level.
"""
import subprocess
import json
from pathlib import Path

import torch
import numpy as np
from model import VAEMario

from level_utils import tensor_to_sim_level, clean_level

Tensor = torch.Tensor

filepath = Path(__file__).parent.resolve()
JARFILE_PATH = f"{filepath}/simulator.jar"


def test_level_from_decoded_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
) -> dict:
    if len(level.shape) < 4:
        level = level.view(1, *level.shape)
    level = tensor_to_sim_level(level)[0]
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def test_level_from_int_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = clean_level(level.detach().numpy())
    level = str(level)

    return run_level(
        level, human_player=human_player, visualize=visualize, max_time=max_time
    )


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


def test_level_from_z(
    z: Tensor, vae: VAEMario, human_player: bool = False, visualize: bool = False
) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results.

    These results are defined in
    simulator.jar <- EvaluationInfo.
    """
    # Get the level from the VAE
    res = vae.decode(z.view(1, -1)).probs.argmax(dim=-1)
    level = res[0]

    return test_level_from_decoded_tensor(
        level, human_player=human_player, visualize=visualize
    )


if __name__ == "__main__":
    human_player = False
    vae = VAEMario()
    vae.load_state_dict(torch.load(filepath / "example.pt"))

    random_z = 3.0 * torch.randn((2,))
    res = test_level_from_z(random_z, vae, human_player=human_player)
    print(res)
