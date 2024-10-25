from pathlib import Path

import numpy as np

from poli.core.data_package import DataPackage


class RFPFoldXStabilitySupervisedDataPackage(DataPackage):
    def __init__(self):
        PROTEIN_DATA_PACKAGES_DIR = Path(__file__).parent
        sequences = np.loadtxt(
            PROTEIN_DATA_PACKAGES_DIR / "rfp_sequences.txt", dtype=str
        )
        rasp_scores = np.loadtxt(PROTEIN_DATA_PACKAGES_DIR / "rfp_foldx_scores.txt")
        padding_token = ""
        max_sequence_length = max(len(sequence) for sequence in sequences)
        unsupervised_data = np.array(
            [
                list(sequence) + [padding_token] * (max_sequence_length - len(sequence))
                for sequence in sequences
            ]
        )
        supervised_data = unsupervised_data, rasp_scores.reshape(-1, 1)

        super().__init__(unsupervised_data, supervised_data)
