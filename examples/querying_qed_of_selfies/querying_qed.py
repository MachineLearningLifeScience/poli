"""
TODO: Write.
"""
from pathlib import Path

import numpy as np

import selfies as sf

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    path_to_alphabet = THIS_DIR / "alphabet_selfies.json"
    problem_info, f, x0, y0, _ = objective_factory.create(name="rdkit_qed", seed=0, path_to_alphabet=path_to_alphabet, string_representation="SELFIES")

    # SELFIES of aspirin
    smiles_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # transforming it to SELFIES
    selfies_aspirin = sf.encoder(smiles_aspirin)
    selfies_aspirin = sf.split_selfies(selfies_aspirin)

    # Using the alphabet to transform it to a sequence of integers
    x_aspirin = np.array([[f.alphabet[c] for c in selfies_aspirin]])
    print(f(x_aspirin))
