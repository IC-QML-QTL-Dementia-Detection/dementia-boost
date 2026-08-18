"""Standardized dual console and timestamped file logger setup.

This module provides `setup_logger` to configure formatted logging streams
that output simultaneously to standard output (sys.stdout) and unique timestamped
log files on disk.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Configures and returns a standardized dual-output logger.

    Streams formatted log records simultaneously to stdout and to a persistent,
    timestamped log file under `log_dir` (e.g., `logs/YYYYMMDD_HHMMSS_{name}.log`).
    If the requested logger already has handlers configured, it is returned
    as-is to avoid duplicate logging.

    Args:
        name: Unique name identifier for the logger instance.
        log_dir: Directory path where timestamped log files will be saved.
            Defaults to "logs".

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    logger_fmt = (
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    )
    formatter = logging.Formatter(
        fmt=logger_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{name}.log")

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
