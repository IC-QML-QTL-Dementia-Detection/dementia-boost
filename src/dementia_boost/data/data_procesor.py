import os


class DataProcessor:
    """
    Handles the ETL process: loads raw OASIS-II dataset,
    processes and filters required pictures.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self) -> None:
        """
        Initializes the processor and ensures the necessary directories exist.
        """
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)

    pass
