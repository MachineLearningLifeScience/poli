from pathlib import Path

import torch

from .inner_rasp.cavity_model import (
    CavityModel,
    DownstreamModel,
)
from .inner_rasp.helpers import init_lin_weights


THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)


def load_cavity_and_downstream_models(DEVICE: str = "cpu"):
    # DEVICE = "cpu"

    # TODO: Ask why this was implemented this way.
    # A transparent alternative would be to simply
    # load the model from the path itself.
    best_cavity_model_path = RASP_DIR / "cavity_model_15.pt"
    cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
    cavity_model_net.load_state_dict(
        torch.load(f"{best_cavity_model_path}", map_location=DEVICE)
    )
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(DEVICE)
    ds_model_net.apply(init_lin_weights)
    ds_model_net.eval()

    return cavity_model_net, ds_model_net

    ...
