"""
This is an example of how to use the RaSP black box
and objective factory.
"""
from pathlib import Path

import numpy as np

from poli.objective_repository.rasp.register import RaspBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residue_strings


THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_file = THIS_DIR / "101m.pdb"

    info = ProblemSetupInformation(
<<<<<<< HEAD
        name="rasp", max_sequence_length=np.inf, alphabet=AMINO_ACIDS, aligned=False
=======
        name="rasp",
        max_sequence_length=np.inf,
        alphabet=AMINO_ACIDS,
        aligned=False
>>>>>>> 437f6cda794042b922ad8513e1902cee879c8419
    )

    f = RaspBlackBox(info=info, wildtype_pdb_path=wildtype_pdb_file)

    clean_wildtype_pdb_file = f.clean_wildtype_pdb_files[0]

    wildtype_residue_string = parse_pdb_as_residue_strings(clean_wildtype_pdb_file)

    # Mutating one position at random
    # to a random amino acid
    x0 = np.array([wildtype_residue_string])
    x = np.copy(x0)
    x[0, np.random.randint(0, x0.shape[1])] = np.random.choice(AMINO_ACIDS)

    # Evaluating on this single mutation
    y = f(x)
