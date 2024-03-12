from poli.core.black_box_information import BlackBoxInformation

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

foldx_rfp_lambo_information = BlackBoxInformation(
    name="foldx_rfp_lambo",
    max_sequence_length=244,
    aligned=False,
    fixed_length=False,
    deterministic=True,  # ?
    alphabet=AMINO_ACIDS,
    discrete=True,
    fidelity=None,
    padding_token="-",
)
