from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with rich console handler and optional file handler."""
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, show_time=True, show_level=True)
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")
    return logging.getLogger("docxru")
