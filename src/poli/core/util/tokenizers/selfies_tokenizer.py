from __future__ import annotations

import selfies as sf

from poli.core.util.tokenizers.abstract_tokenizer import AbstractTokenizer


class SELFIESTokenizer(AbstractTokenizer):
    def __init__(
        self,
        max_sequence_length: int | float | None = None,
        padding_token: str = "[nop]",
    ) -> None:
        super().__init__(max_sequence_length, padding_token)

    def _tokenize(self, texts: str | list[str]) -> list[str] | list[list[str]]:
        if isinstance(texts, str):
            return list(sf.split_selfies(texts))
        elif isinstance(texts, list):
            return [list(sf.split_selfies(t)) for t in texts]
        else:
            raise ValueError(f"Expected str or list, got {type(texts)}")
