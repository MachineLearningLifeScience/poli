from poli.core.black_box_information import BlackBoxInformation

AA = [
    "a",
    "r",
    "n",
    "d",
    "c",
    "q",
    "e",
    "g",
    "h",
    "i",
    "l",
    "k",
    "m",
    "f",
    "p",
    "s",
    "t",
    "w",
    "y",
    "v",
]

gfp_cbas_info = BlackBoxInformation(
    name="gfp_cbas",
    max_sequence_length=237,  # max len of aaSequence
    aligned=True,
    fixed_length=True,
    deterministic=False,
    alphabet=AA,
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)
