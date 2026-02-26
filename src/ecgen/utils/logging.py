import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with a simple, consistent format.
    """

    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
    return logger
