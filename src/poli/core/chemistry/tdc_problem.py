from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from poli.core.chemistry.data_packages import RandomMoleculesDataPackage
from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.problem import Problem

if TYPE_CHECKING:
    from poli.objective_repository.rdkit_logp.register import LogPBlackBox
    from poli.objective_repository.rdkit_qed.register import QEDBlackBox


class TDCProblem(Problem):
    def __init__(
        self,
        black_box: TDCBlackBox | QEDBlackBox | LogPBlackBox,
        x0,
        data_package=None,
        strict_validation=True,
    ):
        if data_package is None:
            data_package = RandomMoleculesDataPackage(
                cast(Literal["SELFIES", "SMILES"], black_box.string_representation)
            )

        super().__init__(
            black_box=black_box,
            x0=x0,
            data_package=data_package,
            strict_validation=strict_validation,
        )
