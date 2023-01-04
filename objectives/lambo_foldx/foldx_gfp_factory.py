import os
import warnings

import numpy as np

from core.AbstractBlackBox import BlackBox
from core.AbstractProblemFactory import ProblemSetupInformation
from core.registry import COMMONS
from objectives.common.lambo.candidate import FoldedCandidate
from objectives.common.lambo.utils import ResidueTokenizer
from objectives.lambo_foldx.abstract_foldx_factory import AbstractFoldXFactory
from objectives.common.lambo.proxy_rfp import ProxyRFPTask
from objectives.common.cbas.util import get_experimental_X_y, convert_aas_to_idx_array, convert_idx_array_to_aas

AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]:i for i in range(len(AA))}


class FoldXGFPFactory(AbstractFoldXFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(name="GFP_FOLDX", max_sequence_length=237, aligned=True, alphabet=AA_IDX)

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        data_path = os.path.join(COMMONS, "data", "cbas_green_fluorescent_protein")
        wt_pdb_file = os.path.join(data_path, "1ema.pdb")
        X, _, _ = get_experimental_X_y(prefix=data_path)

        # parser = PDB.PDBParser()
        # pdb_path = Path(wt_pdb_file).expanduser()
        # struct = parser.get_structure(pdb_path.stem, pdb_path)
        # from Bio.SeqUtils import seq1
        # chain_residues = {
        #     chain.get_id(): seq1(''.join(x.resname for x in chain)) for chain in struct.get_chains()
        # }['A']

        """
        Fasta sequence of the GFP:
        MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
        The sequences in X are missing the initial M.
        """

        X = convert_aas_to_idx_array(X)

        # TODO: REMOVE
        class Bla(BlackBox): pass
        f = Bla(237)
        X = X[:2, :]
        y = read_cached_target_values(f, X)
        return f, X, y

        # parser = PDB.PDBParser()
        # pdb_path = Path(wt_pdb_file).expanduser()
        # struct = parser.get_structure(pdb_path.stem, pdb_path)
        # chain_idxs = {
        #     chain.get_id(): [x.get_id()[1] for x in chain] for chain in struct.get_chains()
        # }
        # assert(len(chain_idxs) == 1)
        # chain_idx = chain_idxs['A']

        # # foldx complains that the following residues are missing
        # incomplete_residues = [6, 26, 52, 101, 107, 122, 124, 131, 132, 156, 157, 158, 162, 212, 214]
        # chain_idx = np.setdiff1d(np.arange(X.shape[1]), incomplete_residues)

        class config: pass  # dummy object to set values
        config.log_dir = "foo"
        config.job_name = "bar"
        config.timestamp = "baz"
        #x0_, y0_, _, _ = task.task_setup(config, project_root=os.getcwd())
        work_dir = f'{os.getcwd()}/{config.log_dir}/{config.job_name}/{config.timestamp}/foldx'
        parent_pdb_path = wt_pdb_file
        tokenizer = ResidueTokenizer()
        chain_id = 'A'
        wild_name = 'gfp'  # Necessary?
        wt = FoldedCandidate(work_dir, parent_pdb_path, [], tokenizer,
                             skip_minimization=True, chain=chain_id, wild_name=wild_name)

        task = ProxyRFPTask(None, None, None, num_start_examples=512)

        missing_pre_sequence = 'M'
        missing_post_sequence = 'THGMDELYK'

        class LamboSasaGFP(BlackBox):
            def _black_box(self, x: np.ndarray) -> np.ndarray:
                """
                There is a mismatch between PDB file and FASTA sequence in position 63 of the PDB.
                It's an X whereas the FASTA sequence has a TYG.
                The PDB file also says that something is odd there.
                Furthermore, the first and the last nine residues are missing.
                """
                #x_ = convert_idx_array_to_aas(x[:, :63])[0].upper() + 'X' + convert_idx_array_to_aas(x[:, 66:-9])[0].upper()  # also remove missing post sequence
                x_ = convert_idx_array_to_aas(x[:, :63])[0].upper() + convert_idx_array_to_aas(x[:, 66:-9])[0].upper()  # also remove missing post sequence

                l = task.make_new_candidates(np.array([wt]), np.array([x_]))
                v = l[0].mutant_total_energy
                return np.array([[v]])

        f = LamboSasaGFP(self.get_setup_information().get_max_sequence_length())
        #y = f(X[:1, :])
        # TODO: use whole dataset
        X = X[:2, :]
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
        y = np.zeros([X.shape[0], 1])
        for i in range(X.shape[0]):
            y[i] = f(X[i:i+1, :])
        np.save(file_name, y)
    return y
