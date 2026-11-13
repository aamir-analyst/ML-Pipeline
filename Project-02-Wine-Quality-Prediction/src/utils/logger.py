
import logging
from pathlib import Path


def setup_logger(name="wine_quality"):

    project_root = Path(__file__).resolve().parents[2]

    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "wine_quality.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # File Handler
        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        # Console Handler
        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
