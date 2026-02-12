"""Wrapper de preflight que reutiliza EnvironmentValidator."""

from __future__ import annotations

from environment_validator import main


if __name__ == "__main__":
    raise SystemExit(main())
