from __future__ import annotations

import numpy as np


def _pad(
    texts: list[list[str]], max_sequence_length: int, padding_token: str = ""
) -> list[list[str]]:
    if max_sequence_length < max(map(len, texts)):
        raise ValueError(
            f"max_sequence_length ({max_sequence_length}) must be greater than the length of the longest text ({max(map(len, texts))})"
        )
    return [
        text + [padding_token] * (max_sequence_length - len(text)) for text in texts
    ]


class AbstractTokenizer:
    def __init__(
        self, max_sequence_length: int | float | None = None, padding_token: str = ""
    ) -> None:
        if isinstance(max_sequence_length, float):
            assert (
                max_sequence_length == np.inf
            ), "max_sequence_length must be np.inf if float"

        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token

    def _tokenize(self, texts: str | list[str]) -> list[str] | list[list[str]]:
        raise NotImplementedError

    def tokenize(self, texts: str | list[str]) -> np.ndarray:
        unpadded_tokens = self._tokenize(texts)

        if isinstance(unpadded_tokens[0], list):
            tokens = unpadded_tokens
        else:
            tokens = [unpadded_tokens]

        if self.max_sequence_length is None or np.isinf(self.max_sequence_length):
            max_length = max(map(len, tokens))
            tokens = _pad(tokens, max_length, padding_token=self.padding_token)
        else:
            tokens = _pad(
                tokens, self.max_sequence_length, padding_token=self.padding_token
            )

        return np.array(tokens)
