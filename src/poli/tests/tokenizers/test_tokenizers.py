import numpy as np
import pytest

from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.tokenizers.character_tokenizer import CharacterTokenizer


def test_character_tokenizer_on_single_sequence():
    one_amino_acid = AMINO_ACIDS[0]

    text = one_amino_acid * 10

    tokenizer = CharacterTokenizer(max_sequence_length=10)

    tokens = tokenizer.tokenize(text)
    assert (tokens == np.array(list(one_amino_acid * 10))).all()


def test_character_tokenizer_on_multiple_sequences_of_same_length():
    one_amino_acid = AMINO_ACIDS[0]

    texts = [one_amino_acid * 10] * 5

    tokenizer = CharacterTokenizer(max_sequence_length=10)

    tokens = tokenizer.tokenize(texts)
    assert (tokens == np.array([list(one_amino_acid * 10)] * 5)).all()


@pytest.mark.parametrize("max_sequence_length", [np.inf, 15, None])
def test_character_tokenizer_on_multiple_sequences_of_varying_length(
    max_sequence_length,
):
    one_amino_acid = AMINO_ACIDS[0]

    texts = [
        one_amino_acid * 10,
        one_amino_acid * 5,
        one_amino_acid * 15,
        one_amino_acid * 3,
        one_amino_acid * 8,
    ]

    tokenizer = CharacterTokenizer(max_sequence_length=max_sequence_length)

    tokens = tokenizer.tokenize(texts)
    assert tokens.shape == (5, 15)
    assert tokens[0, 11] == tokenizer.padding_token


def test_character_tokenizer_outputs_error_on_wrong_max_sequence():
    one_amino_acid = AMINO_ACIDS[0]

    texts = [
        one_amino_acid * 10,
        one_amino_acid * 5,
        one_amino_acid * 15,
        one_amino_acid * 3,
        one_amino_acid * 8,
    ]

    tokenizer = CharacterTokenizer(max_sequence_length=10)

    with pytest.raises(ValueError):
        _ = tokenizer.tokenize(texts)


if __name__ == "__main__":
    test_character_tokenizer_on_multiple_sequences_of_varying_length()
