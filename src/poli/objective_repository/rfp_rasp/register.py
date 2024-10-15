from __future__ import annotations

from pathlib import Path

from poli.core.problem import Problem
from poli.core.proteins.data_packages import RFPRaspSupervisedDataPackage
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.rasp.register import RaspBlackBox, RaspProblemFactory


class RFPRaspBlackBox(RaspBlackBox):
    def __init__(
        self,
        additive=True,
        penalize_unfeasible_with=None,
        device=None,
        experiment_id=None,
        tmp_folder=None,
        batch_size=None,
        parallelize=False,
        num_workers=None,
        evaluation_budget=float("inf"),
        force_isolation=False,
    ):
        RFP_PDB_PATH = Path(__file__).parent / "assets"
        wildtype_pdb_path = [
            RFP_PDB_PATH / "2vad_A.pdb",
            RFP_PDB_PATH / "2vae_D.pdb",
            RFP_PDB_PATH / "3e5v_A.pdb",
            RFP_PDB_PATH / "3ned_A.pdb",
            RFP_PDB_PATH / "5lk4_A.pdb",
            RFP_PDB_PATH / "6aa7_A.pdb",
        ]
        chains_to_keep = ["A", "D", "A", "A", "A", "A"]
        super().__init__(
            wildtype_pdb_path,
            additive,
            chains_to_keep,
            penalize_unfeasible_with,
            device,
            experiment_id,
            tmp_folder,
            batch_size,
            parallelize,
            num_workers,
            evaluation_budget,
            force_isolation,
        )


class RFPRaspProblemFactory(RaspProblemFactory):
    def create(
        self,
        additive: bool = True,
        penalize_unfeasible_with: float | None = None,
        device: str | None = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        rfp_rasp = RFPRaspBlackBox(
            additive=additive,
            penalize_unfeasible_with=penalize_unfeasible_with,
            device=device,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        x0 = rfp_rasp.x0

        data_package = RFPRaspSupervisedDataPackage()

        return Problem(black_box=rfp_rasp, x0=x0, data_package=data_package)
