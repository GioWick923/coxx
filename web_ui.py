"""Entrypoint de UI/API que reexporta servidor modularizado."""

from __future__ import annotations

from api_server import AppState, Handler, MAX_REQUEST_BYTES, STATE, parse_multipart_file, main

__all__ = [
    "AppState",
    "Handler",
    "MAX_REQUEST_BYTES",
    "STATE",
    "parse_multipart_file",
    "main",
]


if __name__ == "__main__":
    main()
