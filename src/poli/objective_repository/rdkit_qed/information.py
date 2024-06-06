import numpy as np

from poli.core.black_box_information import BlackBoxInformation

rdkit_qed_info = BlackBoxInformation(
    name="rdkit_qed",
    max_sequence_length=np.inf,
    aligned=False,
    fixed_length=False,
    deterministic=True,
    alphabet=None,  # TODO: add once we settle for one
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)
