from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.protocols.loops import get_fa_scorefxn
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from poli.core.abstract_isolated_function import AbstractIsolatedFunction

pyrosetta.init()

ROSETTA_ENERGY_DIR = Path(__file__).parent.resolve()


class RosettaEnergyIsolatedLogic(AbstractIsolatedFunction):
    def __init__(self):
        self.mover = MinMover()
        self.energy_function = get_fa_scorefxn()
        self.mover.score_function(self.energy_function)

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        # protein = pyrosetta.pose_from_pdb(ROSETTA_ENERGY_DIR + "/1ggx.pdb")
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
    # from poli.core.registry import register_isolated_function

    # register_isolated_function(
    #     RosettaEnergyIsolatedLogic,  # Your function, designed to be isolated
    #     name=rosetta_energy_information.get_problem_name() + "__isolated",  #  Same name as the problem and folder, ending on __isolated.
    #     conda_environment_name="poli__rosetta_energy",  # The name of the conda env inside environment.yml.
    # )
    isolated_logic = RosettaEnergyIsolatedLogic()
    from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residue_strings

    pdb_path = Path(ROSETTA_ENERGY_DIR) / "1ggx.pdb"
    seq = parse_pdb_as_residue_strings(pdb_path)

    print(seq)
    y = isolated_logic._apply_on_sequence("".join(seq))
    print(y)

    pose = pyrosetta.pose_from_pdb(str(ROSETTA_ENERGY_DIR / "1ggx.pdb"))
    print(pose)
