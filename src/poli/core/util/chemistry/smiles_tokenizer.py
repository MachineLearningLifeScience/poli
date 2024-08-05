from __future__ import annotations

import re

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|
#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

REGEX_FOR_SMILES = re.compile(SMI_REGEX_PATTERN)


def tokenize_smiles(smiles: str | list[str]) -> list[str] | list[list[str]]:
    """
    Tokenize a SMILES strings using the Basic tokenizer from DeepChem [1].
    """

    if isinstance(smiles, str):
        smiles = [smiles]

    tokens = [list(REGEX_FOR_SMILES.findall(smile)) for smile in smiles]

    # Make sure they all have the same length
    max_len = max(len(token) for token in tokens)

    for token in tokens:
        token += [""] * (max_len - len(token))

    if len(tokens) == 1:
        return tokens[0]

    return tokens
