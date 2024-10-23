import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.proteins.defaults import AMINO_ACIDS

rosetta_energy_information = BlackBoxInformation(
    name="rosetta_energy",
    max_sequence_length=np.inf,
    alphabet=AMINO_ACIDS,
    aligned=True,
    fixed_length=True,
    discrete=True,
    deterministic=False,
    padding_token="",
)
