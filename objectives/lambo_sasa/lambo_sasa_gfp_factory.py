import numpy as np

from core.AbstractBlackBox import BlackBox
from core.AbstractProblemFactory import AbstractProblemFactory, ProblemSetupInformation
from objectives.lambo_sasa.sasa import SurfaceArea


class LamboSasaGFPFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        pass

    def create(self) -> (BlackBox, np.ndarray, np.ndarray):
        sasa = SurfaceArea()
        # TODO: load pdb file

        class LamboSasaGFP(BlackBox):
            def _black_box(self, x: np.ndarray) -> np.ndarray:
                # TODO: derive mutant pdb file
                y = sasa(name=uuid, loc=mutant_pdb_file)
                return np.array([[y]])

        f = LamboSasaGFP()
        return f