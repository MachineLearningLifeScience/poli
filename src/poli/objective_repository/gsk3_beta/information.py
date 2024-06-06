import numpy as np

from poli.core.black_box_information import BlackBoxInformation

gsk3_beta_info = BlackBoxInformation(
    name="gsk3_beta",
    max_sequence_length=np.inf,
    aligned=False,
    fixed_length=False,
    deterministic=True,  # ?
    alphabet=None,  # TODO: add alphabet once we settle for one for SMLIES/SELFIES.
    log_transform_recommended=False,
    discrete=True,
    padding_token="",
)
