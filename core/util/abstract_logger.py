class AbstractLogger:
    def log(self, metrics: dict, step: int) -> None:
        raise NotImplementedError("abstract method")
