from src.poli.core.problem_setup_information import ProblemSetupInformation


class AbstractLogger:
    def log(self, metrics: dict, step: int) -> None:
        raise NotImplementedError("abstract method")

    def initialize_logger(self, problem_setup_info: ProblemSetupInformation, caller_info) -> str:
        raise NotImplementedError("abstract method")

    def finish(self) -> None:
        raise NotImplementedError("abstract method")
