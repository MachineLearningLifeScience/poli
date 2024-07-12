import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.proteins.defaults import AMINO_ACIDS

rmf_info = BlackBoxInformation(
    name="rmf_landscape",
    max_sequence_length=np.inf,
    aligned=True,
    fixed_length=True,
    deterministic=False,
    alphabet=AMINO_ACIDS,  # TODO: differentiate between AA and NA inputs?
    log_transform_recommended=False,
    discrete=True,
)
