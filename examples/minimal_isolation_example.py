import numpy as np

from poli.repository import EhrlichHoloBlackBox

f = EhrlichHoloBlackBox(
    sequence_length=10,
    motif_length=3,
    n_motifs=2,
)

print(f.alphabet)
print(f)
print(f(np.array(["ACGTACGTAA", "ACGTACGTAC"])))
print(f)
