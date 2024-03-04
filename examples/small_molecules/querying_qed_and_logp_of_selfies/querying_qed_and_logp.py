"""
This is a small example of how to compute the
QED and logP of a SELFIES string using poli.

To load an objective function for QED or logP,
you will need to specify the path to the alphabet
(i.e. the dictionary {token (str): id (int)}). This
is because the alphabet is not hardcoded in poli, since
different problems/datasets will have different alphabets.
"""

from pathlib import Path

import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    f_qed, x0, y0 = objective_factory.create(
        name="rdkit_qed",
        string_representation="SELFIES",
    )

    f_logp, x0, y0 = objective_factory.create(
        name="rdkit_logp",
        string_representation="SELFIES",
    )

    # SELFIES of aspirin
    selfies_aspirin = np.array(
        [
            [
                "[C]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[Ring1]",
                "[=Branch1]",
                "[C]",
                "[=Branch1]",
                "[C]",
                "[=O]",
                "[O]",
            ]
        ]
    )

    print(f"QED of aspirin: {f_qed(selfies_aspirin)}")
    print(f"logP of aspirin: {f_logp(selfies_aspirin)}")
