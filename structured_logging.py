"""Structured logging utilities for backend services."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python log records as JSON with contextual metadata."""

    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts_ms": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _normalize_level(value: str) -> int:
    return getattr(logging, (value or "INFO").upper(), logging.INFO)


def get_structured_logger(name: str) -> logging.Logger:
    """Return a logger configured for JSON console + rotating file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = _normalize_level(os.getenv("RAG_LOG_LEVEL", "INFO"))
    log_file = os.getenv("RAG_LOG_FILE", "./logs/rag_backend.log")
    max_bytes = int(os.getenv("RAG_LOG_MAX_BYTES", "1048576"))
    backup_count = int(os.getenv("RAG_LOG_BACKUP_COUNT", "5"))

    logger.setLevel(level)
    formatter = StructuredJsonFormatter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rotating = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    rotating.setLevel(level)
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)

    logger.propagate = False
    return logger
