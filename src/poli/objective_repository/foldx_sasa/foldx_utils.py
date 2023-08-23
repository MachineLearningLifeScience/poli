AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
MAX_SEQUENCE_LENGTH = 100

ENCODING = {amino_acid: i for i, amino_acid in enumerate(AMINO_ACIDS)}
INVERSE_ENCODING = {i: amino_acid for amino_acid, i in ENCODING.items()}
