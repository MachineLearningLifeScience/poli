from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.proteins.defaults import AMINO_ACIDS

gfp_select_info = BlackBoxInformation(
    name="gfp_select",
    max_sequence_length=237,  # max len of aaSequence
    aligned=True,  # TODO: perhaps add the fact that there is a random state here?
    fixed_length=True,
    deterministic=False,
    alphabet=AMINO_ACIDS,
    log_transform_recommended=False,
    discrete=True,
    fidelity=None,
    padding_token="",
)
