"""Preflight de entorno para ejecutar RAG Studio Pro localmente."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "bs4",
    "langchain_community",
    "langchain_core",
    "langchain_text_splitters",
    "langchain_ollama",
    "chromadb",
    "pypdf",
]

REQUIRED_FILES = [
    Path("web_ui.py"),
    Path("rag_mejorado.py"),
    Path("requirements.txt"),
    Path("assets/cyber_bg.svg"),
]


def check_python() -> list[str]:
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append("Python 3.10+ es requerido")
    return issues


def check_modules() -> list[str]:
    issues: list[str] = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            issues.append(f"Módulo faltante: {module}")
    return issues


def check_files_and_dirs() -> list[str]:
    issues: list[str] = []
    for file_path in REQUIRED_FILES:
        if not file_path.exists():
            issues.append(f"Archivo faltante: {file_path}")

    # Carpeta data por defecto para ingesta local.
    Path("data").mkdir(parents=True, exist_ok=True)
    return issues


def check_ollama() -> list[str]:
    issues: list[str] = []
    if shutil.which("ollama") is None:
        return ["No se encontró el comando 'ollama' en PATH"]

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            issues.append("`ollama list` falló; verifica que Ollama esté ejecutándose")
            return issues

        output = result.stdout.lower()
        if "nomic-embed-text" not in output:
            issues.append("Falta modelo de embeddings: nomic-embed-text")
        if "llama3.1:8b" not in output:
            issues.append("Falta modelo de chat: llama3.1:8b")
    except Exception as exc:
        issues.append(f"No se pudo verificar Ollama: {exc}")

    return issues


def main() -> int:
    checks = {
        "python": check_python(),
        "modules": check_modules(),
        "files": check_files_and_dirs(),
        "ollama": check_ollama(),
    }

    has_issues = False
    print("== Preflight RAG Studio Pro ==")
    for section, issues in checks.items():
        if issues:
            has_issues = True
            print(f"[WARN] {section}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {section}")

    print("\nSugerencia: si hay WARN de módulos, ejecuta `pip install -r requirements.txt`.")
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
