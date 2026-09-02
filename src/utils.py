"""
Utility functions: logging setup, timing decorators, helpers.
"""

import logging
import time
import functools
from typing import Callable, Any


def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def timer(func: Callable) -> Callable:
    """Decorator that logs the wall-clock time of a function call."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger("pipeline")
        logger.info(f"▶ Starting {func.__name__}...")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"✓ {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper
