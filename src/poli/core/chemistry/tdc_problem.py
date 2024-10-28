from poli.core.chemistry.data_packages import RandomMoleculesDataPackage
from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.problem import Problem


class TDCProblem(Problem):
    def __init__(
        self, black_box: TDCBlackBox, x0, data_package=None, strict_validation=True
    ):
        if data_package is None:
            data_package = RandomMoleculesDataPackage(black_box.string_representation)

        super().__init__(
            black_box=black_box,
            x0=x0,
            data_package=data_package,
            strict_validation=strict_validation,
        )
