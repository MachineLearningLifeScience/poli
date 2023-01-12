import numpy as np
import os

from src.poli.core import BlackBox
from src.poli.core import abstract_problem_factory, ProblemSetupInformation
from src.poli.objectives.common import __file__ as common_path


AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]:i for i in range(len(AA))}


class LamboSasaGFPFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation(name="GFP_SASA", max_sequence_length=237, aligned=True, alphabet=AA_IDX)

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        from src.poli.objectives.common.cbas.util import get_experimental_X_y
        from src.poli.objectives.lambo_sasa.sasa import SurfaceArea

        sasa = SurfaceArea()
        wt_pdb_file = os.path.join(os.path.dirname(common_path), "data", "cbas_green_fluorescent_protein", "1ema.pdb")
        uuid = "foobar"

        class LamboSasaGFP(BlackBox):
            def _black_box(self, x: np.ndarray) -> np.ndarray:
                # TODO: derive mutant pdb file
                raise NotImplementedError("complete implementation")
                s = sasa(name=uuid, loc=mutant_pdb_file)
                return np.array([[s]])

        f = LamboSasaGFP(self.get_setup_information().get_max_sequence_length())
        X, _, _ = get_experimental_X_y()
        y = f(X)
        return f, X, y
