from pathlib import Path

import numpy as np

from poli.core.data_package import DataPackage


class RFPRaspSupervisedDataPackage(DataPackage):
    def __init__(self):
        PROTEIN_DATA_PACKAGES_DIR = Path(__file__).parent
        sequences = np.loadtxt(
            PROTEIN_DATA_PACKAGES_DIR / "rfp_sequences.txt", dtype=str
        )
        rasp_scores = np.loadtxt(PROTEIN_DATA_PACKAGES_DIR / "rfp_rasp_scores.txt")
        unsupervised_data = sequences
        supervised_data = sequences, rasp_scores

        super().__init__(unsupervised_data, supervised_data)
