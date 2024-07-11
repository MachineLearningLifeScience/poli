import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.proteins.defaults import AMINO_ACIDS

foldx_stability_and_sasa_info = BlackBoxInformation(
    name="foldx_stability_and_sasa",
    max_sequence_length=np.inf,
    aligned=False,
    fixed_length=False,
    deterministic=True,
    alphabet=AMINO_ACIDS,
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)
