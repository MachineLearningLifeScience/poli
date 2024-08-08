from __future__ import annotations

from poli.core.util.tokenizers.abstract_tokenizer import AbstractTokenizer


class ListTokenizer(AbstractTokenizer):
    def _tokenize(self, texts: str | list[str]) -> list[str] | list[list[str]]:
        if isinstance(texts, str):
            return list(texts)
        elif isinstance(texts, list):
            return [list(t) for t in texts]
        else:
            raise ValueError(f"Expected str or list, got {type(texts)}")
