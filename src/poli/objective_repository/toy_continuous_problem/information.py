import numpy as np

from poli.core.black_box_information import BlackBoxInformation

toy_continuous_info = BlackBoxInformation(
    name="toy_continuous_problem",
    max_sequence_length=np.inf,
    aligned=True,
    fixed_length=True,
    deterministic=True,
    alphabet=None,
    log_transform_recommended=False,
    discrete=False,
    padding_token=None,
)
