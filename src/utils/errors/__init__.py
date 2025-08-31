class MissingEnvironmentVariableError(Exception): ...


class DatasetNotImplementedError(NotImplementedError):
    """Exception raised when a dataset is not implemented."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' is not implemented")
