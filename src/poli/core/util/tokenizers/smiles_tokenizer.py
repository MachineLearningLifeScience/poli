from __future__ import annotations

import re

from poli.core.util.tokenizers.abstract_tokenizer import AbstractTokenizer


class SMILESTokenizer(AbstractTokenizer):
    def __init__(
        self, max_sequence_length: int | float | None = None, padding_token: str = ""
    ) -> None:
        # DeepChem's SMILES tokenizer plus tokenizing single integers
        self.REGEX_FOR_SMILES = re.compile(
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\d{1})"
        )
        super().__init__(max_sequence_length, padding_token)

    def _tokenize(self, texts: str | list[str]) -> list[str] | list[list[str]]:
        smiles = texts
        if isinstance(smiles, str):
            smiles = [smiles]

        tokens = [list(self.REGEX_FOR_SMILES.findall(smile)) for smile in smiles]

        # Make sure they all have the same length
        max_len = max(len(token) for token in tokens)

        for token in tokens:
            token += [""] * (max_len - len(token))

        if len(tokens) == 1:
            return tokens[0]

        return tokens
