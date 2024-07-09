import os

import numpy as np
import pyrosetta
from pyrosetta.rosetta.protocols.loops import get_fa_scorefxn
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.objective_repository.rosetta_energy import rosetta_energy_information


pyrosetta.init()

ROSETTA_ENERGY_DIR = os.path.dirname(__file__)


class RosettaEnergyIsolatedLogic(AbstractIsolatedFunction):
    def __init__(self):
        self.mover = MinMover()
        self.energy_function = get_fa_scorefxn()
        self.mover.score_function(self.energy_function)


    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        #protein = pyrosetta.pose_from_pdb(ROSETTA_ENERGY_DIR + "/1ggx.pdb")
        y = np.empty([x.shape[0], 1])
        for i in range(x.shape[0]):
            seq = "".join(x[i, :].tolist())
            y[i, 0] = self._apply_on_sequence(seq)
        return y

    def _apply_on_sequence(self, seq: str):
        protein = pyrosetta.pose_from_sequence(seq)
        self.mover.apply(protein)
        return self.energy_function.score(protein)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RosettaEnergyIsolatedLogic,  # Your function, designed to be isolated
        name=rosetta_energy_information.get_problem_name() + "__isolated",  #  Same name as the problem and folder, ending on __isolated.
        conda_environment_name="poli__rosetta_energy",  # The name of the conda env inside environment.yml.
    )
