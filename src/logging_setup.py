import logging
import os
from datetime import datetime

LOGGER_NAME = "nlp_classification"


def setup_logging(model_name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Настраивает общий логгер проекта: пишет одновременно в файл
    logs/{model_name}_{timestamp}.log и в консоль.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{model_name}_{timestamp}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.log_path = log_path  
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
