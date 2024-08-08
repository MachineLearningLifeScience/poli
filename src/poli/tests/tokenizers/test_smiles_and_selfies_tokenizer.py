from __future__ import annotations

import numpy as np
import pytest

from poli.core.util.tokenizers.selfies_tokenizer import SELFIESTokenizer
from poli.core.util.tokenizers.smiles_tokenizer import SMILESTokenizer


@pytest.mark.parametrize(
    "smile, expected_tokens",
    [
        ("CCO", ["C", "C", "O"]),
        ("C1=CC=CC=C1", ["C", "1", "=", "C", "C", "=", "C", "C", "=", "C", "1"]),
        # TODO: add more
    ],
)
def test_smiles_tokenizer_on_single_sequence(smile: str, expected_tokens: list[str]):
    tokenizer = SMILESTokenizer(max_sequence_length=np.inf)

    tokens = tokenizer.tokenize(smile)
    assert (tokens == np.array(expected_tokens)).all()


@pytest.mark.parametrize(
    "selfie, expected_tokens",
    [
        ("[C][C][O]", ["[C]", "[C]", "[O]"]),
        (
            "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
            ["[C]", "[=C]", "[C]", "[=C]", "[C]", "[=C]", "[Ring1]", "[=Branch1]"],
        ),
    ],
)
def test_selfies_tokenizer_on_single_sequence(selfie: str, expected_tokens: list[str]):
    tokenizer = SELFIESTokenizer(max_sequence_length=np.inf)

    tokens = tokenizer.tokenize(selfie)
    assert (tokens == np.array(expected_tokens)).all()


def test_smiles_on_sequences_of_varying_length():
    smiles = [
        "CCO",
        "C1=CC=CC=C1",
        "CC(C)C",
        "CC(C)(C)C",
        "CC(C)(C)C",
    ]

    tokenizer = SMILESTokenizer(max_sequence_length=np.inf)

    tokens = tokenizer.tokenize(smiles)
    assert tokens.shape == (5, 11)
    assert tokens[0, 3] == tokenizer.padding_token
    assert tokens[4, 9] == tokenizer.padding_token


def test_selfies_on_sequences_of_varying_length():
    selfies = [
        "[C][C][O]",
        "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    ]

    tokenizer = SELFIESTokenizer(max_sequence_length=np.inf)

    tokens = tokenizer.tokenize(selfies)
    assert tokens.shape == (2, 8)
    assert tokens[0, 4] == tokenizer.padding_token
