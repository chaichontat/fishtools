import sys

from loguru import logger


def setup_logging() -> None:
    """Configure the concise console logger used by the command-line interface."""
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: >8}</level> | "
            "<cyan>{name}</cyan> | <level>{message}</level>"
        ),
    )

