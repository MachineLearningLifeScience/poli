import numpy as np

from poli.core.black_box_information import BlackBoxInformation

penalized_logp_lambo_info = BlackBoxInformation(
    name="penalized_logp_lambo",
    max_sequence_length=np.inf,
    aligned=False,
    fixed_length=False,
    alphabet=None,  # TODO: add when we settle for an alphabet
    deterministic=True,
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)
