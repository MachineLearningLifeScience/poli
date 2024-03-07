import numpy as np

from poli.core.black_box_information import BlackBoxInformation

from poli.core.util.proteins.defaults import AMINO_ACIDS

rasp_information = BlackBoxInformation(
    name="rasp",
    max_sequence_length=np.inf,
    aligned=True,
    fixed_length=False,
    deterministic=True,
    alphabet=AMINO_ACIDS,
    log_transform_recommended=False,
    discrete=True,
    fidelity="low",
    padding_token="",
)
