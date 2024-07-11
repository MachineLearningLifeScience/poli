import numpy as np

from poli.core.black_box_information import BlackBoxInformation

ehrlich_info = BlackBoxInformation(
    name="ehrlich",
    max_sequence_length=np.inf,
    aligned=True,
    fixed_length=True,
    deterministic=True,  # ?
    alphabet=None,  # TODO: add alphabet once we settle for one for SMLIES/SELFIES.
    log_transform_recommended=False,
    discrete=True,
    padding_token="",
)
