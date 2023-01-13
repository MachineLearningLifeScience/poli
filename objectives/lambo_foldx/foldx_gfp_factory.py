import os
import warnings
import numpy as np
import poli.core.registry

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

# TODO: fix this permanently
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add subfolder to path
from common.lambo.candidate import FoldedCandidate
from common.lambo.utils import ResidueTokenizer
from common.lambo.proxy_rfp import ProxyRFPTask
from common.cbas.util import get_experimental_X_y, convert_aas_to_idx_array, convert_idx_array_to_aas, AA_IDX


class FoldXGFPFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(name="GFP_FOLDX", max_sequence_length=237, aligned=True, alphabet=AA_IDX)

    def create(self, seed: int = 0) -> (AbstractBlackBox, np.ndarray, np.ndarray):
        from common import __file__ as common_file
        COMMONS = os.path.dirname(common_file)
        data_path = os.path.join(COMMONS, "data", "cbas_green_fluorescent_protein")
        wt_pdb_file = os.path.join(data_path, "1ema.pdb")
        X, _, _ = get_experimental_X_y(prefix=data_path, random_state=seed)
        """
        Fasta sequence of the GFP:
        MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
        The sequences in X are missing the initial M.
        """
        X = convert_aas_to_idx_array(X)

        work_dir = os.path.join(os.getcwd(), "temp", "foldx")
        wt = FoldedCandidate(work_dir, wt_pdb_file, [], ResidueTokenizer(), skip_minimization=True, chain='A', wild_name='gfp')
        task = ProxyRFPTask(None, None, None, num_start_examples=512)
        wt_array = np.array([wt])
        x_array = np.array([" " * self.get_setup_information().get_max_sequence_length()])

        class LamboSasaGFP(AbstractBlackBox):
            def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
                x_ = convert_idx_array_to_aas(x[:, :63])[0].upper() + convert_idx_array_to_aas(x[:, 66:-9])[0].upper()
                """
                FoldX complains that the following residues are missing:
                # incomplete_residues = [6, 26, 52, 101, 107, 122, 124, 131, 132, 156, 157, 158, 162, 212, 214]
                But this seems not to be the issue.
                The difference between parsed PDB file and FASTA sequence stems from also a missing 'M' in the beginning
                 and a missing post sequence:
                missing_post_sequence = 'THGMDELYK'
                The authors of the PDB file write something about how this last part was not modelled.
                Furthermore, there is a mismatch between PDB file and FASTA sequence in position 63 of the PDB.
                It's an X whereas the FASTA sequence has a TYG.
                The PDB file also says that something is odd there.
                """
                x_array[0] = x_
                l = task.make_new_candidates(wt_array, x_array)
                v = -l[0].mutant_total_energy  # we are minimizing, hence the minus
                return np.array([[v]])

        f = LamboSasaGFP(self.get_setup_information().get_max_sequence_length())
        # TODO: REMOVE
        X = X[:5, :]
        y = read_cached_target_values(f, X)
        return f, X, y


def read_cached_target_values(f, X):
    file_name = os.path.join(os.path.dirname(__file__), "cached_foldx_values.npy")
    try:
        y = np.load(file_name)
        if not y.shape[0] == X.shape[0]:
            # recompute cache if something is odd
            raise FileNotFoundError()
    except FileNotFoundError:
        warnings.warn("Recomputing cache. This will take a while.")
        from time import time
        y = np.zeros([X.shape[0], 1])
        for i in range(X.shape[0]):
            print(f"iteration: {i}/{X.shape[0]}")
            t = time()
            y[i] = f(X[i:i+1, :])
            print("function evaluation took: " + str(time() - t) + " seconds")
        np.save(file_name, y)
    return y


if __name__ == '__main__':
    poli.core.registry.register_problem(FoldXGFPFactory(), os.path.join(os.path.dirname(__file__), 'foldx_gfp.sh'))