"""
This is a small example of how to compute the
QED and logP of a SELFIES string using poli.

To load an objective function for QED or logP,
you will need to specify the path to the alphabet
(i.e. the dictionary {token (str): id (int)}). This
is because the alphabet is not hardcoded in poli, since
different problems/datasets will have different alphabets.

You will also need to install selfies, which allows you
to encode and decode SELFIES strings, as well as transforming
SELFIES strings into a list of tokens.

Run:

```
pip install selfies
```
"""
from pathlib import Path

import numpy as np

import selfies as sf

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    path_to_alphabet = THIS_DIR / "alphabet_selfies.json"
    problem_info, f_qed, x0, y0, _ = objective_factory.create(
        name="rdkit_qed",
        seed=0,
        path_to_alphabet=path_to_alphabet,
        string_representation="SELFIES",
    )

    _, f_logp, x0, y0, _ = objective_factory.create(
        name="rdkit_logp",
        seed=0,
        path_to_alphabet=path_to_alphabet,
        string_representation="SELFIES",
    )

    # SELFIES of aspirin
    smiles_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # transforming it to SELFIES
    selfies_aspirin = sf.encoder(smiles_aspirin)
    selfies_aspirin = sf.split_selfies(selfies_aspirin)

    # Using the alphabet to transform it to a sequence of integers
    x_aspirin = np.array([[f_qed.alphabet[c] for c in selfies_aspirin]])

    print(f"QED of aspirin: {f_qed(x_aspirin)}")
    print(f"logP of aspirin: {f_logp(x_aspirin)}")
