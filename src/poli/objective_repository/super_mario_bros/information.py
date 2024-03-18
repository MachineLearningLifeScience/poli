from poli.core.black_box_information import BlackBoxInformation

SMB_ALPHABET = [
    "X",
    "S",
    "-",
    "?",
    "Q",
    "E",
    "<",
    ">",
    "[",
    "]",
    "o",
]

smb_info = BlackBoxInformation(
    name="super_mario_bros",
    max_sequence_length=14 * 14,
    aligned=True,
    fixed_length=True,
    deterministic=False,
    alphabet=SMB_ALPHABET,
    log_transform_recommended=True,
    discrete=True,
    padding_token=None,
)
