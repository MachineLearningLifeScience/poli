from pathlib import Path

import numpy as np

from poli.core.data_package import DataPackage


class RFPFoldXStabilitySupervisedDataPackage(DataPackage):
    def __init__(self):
        PROTEIN_DATA_PACKAGES_DIR = Path(__file__).parent
        sequences = np.loadtxt(PROTEIN_DATA_PACKAGES_DIR / "rfp_sequences.txt")
        rasp_scores = np.loadtxt(PROTEIN_DATA_PACKAGES_DIR / "rfp_foldx_scores.txt")
        unsupervised_data = sequences
        supervised_data = sequences, rasp_scores

        super().__init__(unsupervised_data, supervised_data)
