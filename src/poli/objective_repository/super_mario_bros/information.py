from poli.core.black_box_information import BlackBoxInformation

smb_info = BlackBoxInformation(
    name="super_mario_bros",
    max_sequence_length=2,
    aligned=True,
    fixed_length=True,
    deterministic=False,
    alphabet=None,
    log_transform_recommended=True,
    discrete=False,
    padding_token=None,
)
