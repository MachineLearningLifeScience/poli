"""Utilities for loading up the cavity and downstream models for RaSP."""

from pathlib import Path
from typing import Tuple

import torch

from poli.core.util.proteins.rasp.inner_rasp.cavity_model import (
    CavityModel,
    DownstreamModel,
)
from poli.core.util.proteins.rasp.inner_rasp.helpers import init_lin_weights

THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)


def load_cavity_and_downstream_models(
    device: str = "cpu",
) -> Tuple[CavityModel, DownstreamModel]:
    """Load the cavity and downstream models for RaSP.

    Parameters
    -----------
    device : str, optional
        The device to load the models on. Defaults to "cpu".

    Returns
    --------
    cavity_model_net : CavityModel
        The cavity model.
    ds_model_net : DownstreamModel
        The downstream model.
    """
    # DEVICE = "cpu"

    # TODO: Ask why this was implemented this way (i.e.
    # loading from a "best_cavity_model_path").
    # A transparent alternative would be to simply
    # load the model from the path itself.
    best_cavity_model_path = RASP_DIR / "cavity_model_15.pt"
    cavity_model_net = CavityModel(get_latent=True).to(device)
    cavity_model_net.load_state_dict(
        torch.load(f"{best_cavity_model_path}", map_location=device, weights_only=True)
    )
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(device)
    ds_model_net.apply(init_lin_weights)
    ds_model_net.eval()

    return cavity_model_net, ds_model_net
