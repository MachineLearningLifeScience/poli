import numpy as np

from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.black_box_information import BlackBoxInformation

foldx_stability_info = BlackBoxInformation(
    name="foldx_stability",
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
