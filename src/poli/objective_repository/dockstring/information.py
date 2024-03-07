import numpy as np

from poli.core.black_box_information import BlackBoxInformation

dockstring_black_box_information = BlackBoxInformation(
    name="dockstring",
    max_sequence_length=np.inf,
    aligned=False,
    fixed_length=False,
    deterministic=True,
    alphabet=None,  # TODO: fix when we have a smiles alphabet
    log_transform_recommended=False,
    discrete=True,
    padding_token="",
)
