from pathlib import Path

from poli.core.problem import Problem
from poli.core.proteins.data_packages import RFPFoldXStabilitySupervisedDataPackage
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.foldx_stability.register import (
    FoldXStabilityBlackBox,
    FoldXStabilityProblemFactory,
)


class RFPFoldXStabilityBlackBox(FoldXStabilityBlackBox):
    def __init__(
        self,
        experiment_id=None,
        tmp_folder=None,
        eager_repair=False,
        verbose=False,
        batch_size=1,
        parallelize=False,
        num_workers=None,
        evaluation_budget=float("inf"),
        force_isolation=False,
    ):
        RFP_FOLDX_ASSETS_DIR = Path(__file__).parent / "assets"

        wildtype_pdb_path = [
            RFP_FOLDX_ASSETS_DIR / "2vad_A_Repair.pdb",
            RFP_FOLDX_ASSETS_DIR / "2vae_D_Repair.pdb",
            RFP_FOLDX_ASSETS_DIR / "3e5v_A_Repair.pdb",
            RFP_FOLDX_ASSETS_DIR / "3ned_A_Repair.pdb",
            RFP_FOLDX_ASSETS_DIR / "5lk4_A_Repair.pdb",
            RFP_FOLDX_ASSETS_DIR / "6aa7_A_Repair.pdb",
        ]
        super().__init__(
            wildtype_pdb_path,
            experiment_id,
            tmp_folder,
            eager_repair,
            verbose,
            batch_size,
            parallelize,
            num_workers,
            evaluation_budget,
            force_isolation,
        )


class RFPFoldXStabilityProblemFactory(FoldXStabilityProblemFactory):
    def create(
        self,
        experiment_id=None,
        tmp_folder=None,
        eager_repair=False,
        verbose=False,
        seed=None,
        batch_size=1,
        parallelize=False,
        num_workers=None,
        evaluation_budget=float("inf"),
        force_isolation=False,
    ):
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        rfp_foldx = RFPFoldXStabilityBlackBox(
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        x0 = rfp_foldx.x0

        data_package = RFPFoldXStabilitySupervisedDataPackage()

        return Problem(
            black_box=rfp_foldx,
            x0=x0,
            data_package=data_package,
        )
