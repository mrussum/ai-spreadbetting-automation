"""Centralised logging configuration for the trading system."""

import logging
import logging.handlers
import sys
from pathlib import Path

from pythonjsonlogger import jsonlogger

from config.settings import LOG_DIR, LOG_LEVEL


def configure_logging(
    log_to_file: bool = True,
    log_to_stdout: bool = True,
    json_format: bool = False,
) -> None:
    """Configure root logger with consistent format across all modules.

    Call once at application startup (scheduler, dashboard, CLI scripts).

    Args:
        log_to_file: if True, write to LOG_DIR/trading.log with daily rotation.
        log_to_stdout: if True, also stream to stdout.
        json_format: if True, emit JSON log lines (for log aggregators);
                     otherwise use a human-readable format.
    """
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by previous calls or third-party imports
    root.handlers.clear()

    handlers: list[logging.Handler] = []

    if log_to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        handlers.append(stream_handler)

    if log_to_file:
        log_path: Path = LOG_DIR / "trading.log"
        # Rotate daily, keep 14 days of history
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            backupCount=14,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        handlers.append(file_handler)

    if json_format:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s, file=%s, json=%s",
        LOG_LEVEL, log_to_file, json_format,
    )
