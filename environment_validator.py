"""Validador de entorno para RAG Studio Pro.

Este módulo concentra validaciones de prerequisitos para ejecución local:
- Python mínimo
- Dependencias Python
- Ollama instalado/accesible + modelos disponibles
- Acceso de escritura a disco para rutas de trabajo
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config_manager import CONFIG


@dataclass
class CheckResult:
    """Resultado de una validación puntual."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class EnvironmentValidator:
    """Valida requisitos de entorno para ejecutar el sistema RAG."""

    REQUIRED_MODULES: Sequence[str] = (
        "bs4",
        "langchain_community",
        "langchain_core",
        "langchain_text_splitters",
        "langchain_ollama",
        "chromadb",
        "pypdf",
        "requests",
    )

    def __init__(
        self,
        min_python: tuple[int, int] = (3, 10),
        required_models: Sequence[str] | None = None,
    ) -> None:
        cfg = CONFIG.get()
        self.min_python = min_python
        self.required_models = list(required_models or [cfg.rag_embed_model, cfg.rag_chat_model])
        self.storage_paths = [
            Path(cfg.rag_data_dir),
            Path(cfg.rag_chroma_dir),
            Path(cfg.rag_log_file).parent,
        ]

    def validate_python(self) -> CheckResult:
        """Valida versión mínima de Python."""
        current = sys.version_info
        details = {"current": f"{current.major}.{current.minor}.{current.micro}", "minimum": f"{self.min_python[0]}.{self.min_python[1]}"}
        if (current.major, current.minor) < self.min_python:
            return CheckResult(
                ok=False,
                errors=[f"Python {self.min_python[0]}.{self.min_python[1]}+ es requerido."],
                suggestions=["Instala una versión de Python compatible y vuelve a ejecutar el sistema."],
                details=details,
            )
        return CheckResult(ok=True, details=details)

    def validate_dependencies(self) -> CheckResult:
        """Valida dependencias Python requeridas."""
        missing: List[str] = []
        for module_name in self.REQUIRED_MODULES:
            try:
                importlib.import_module(module_name)
            except Exception:
                missing.append(module_name)

        if missing:
            return CheckResult(
                ok=False,
                errors=[f"Dependencias faltantes: {', '.join(missing)}"],
                suggestions=["Ejecuta `pip install -r requirements.txt` o `uv sync` para instalar dependencias."],
                details={"missing_modules": missing},
            )

        return CheckResult(ok=True, details={"checked_modules": list(self.REQUIRED_MODULES)})

    def _parse_ollama_models(self, raw_output: str) -> List[str]:
        models: List[str] = []
        for line in raw_output.splitlines():
            row = line.strip()
            if not row or row.lower().startswith("name"):
                continue
            model_name = row.split()[0]
            if model_name and model_name not in models:
                models.append(model_name)
        return models

    def validate_ollama(self) -> CheckResult:
        """Valida comando `ollama`, conectividad y modelos disponibles."""
        if shutil.which("ollama") is None:
            return CheckResult(
                ok=False,
                errors=["No se encontró el comando `ollama` en PATH."],
                suggestions=["Instala Ollama y verifica que `ollama` esté disponible desde la terminal."],
                details={"detected_models": []},
            )

        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception as exc:
            return CheckResult(
                ok=False,
                errors=[f"No se pudo ejecutar `ollama list`: {exc}"],
                suggestions=["Verifica que Ollama esté iniciado (`ollama serve`) y accesible localmente."],
                details={"detected_models": []},
            )

        if proc.returncode != 0:
            return CheckResult(
                ok=False,
                errors=[f"`ollama list` falló con código {proc.returncode}."],
                warnings=[proc.stderr.strip() or "Sin detalle adicional de stderr."],
                suggestions=["Asegura que el servicio de Ollama esté activo e intenta nuevamente."],
                details={"detected_models": []},
            )

        models = self._parse_ollama_models(proc.stdout)
        missing_required = [m for m in self.required_models if m and m not in models]

        warnings: List[str] = []
        suggestions: List[str] = []
        if not models:
            warnings.append("Ollama está disponible pero no hay modelos instalados.")
            suggestions.append("Descarga modelos con `ollama pull <modelo>`, por ejemplo `ollama pull llama3.1:8b`.")

        if missing_required:
            warnings.append(f"Modelos configurados no encontrados: {', '.join(missing_required)}")
            suggestions.append("Actualiza perfiles/modelos activos desde la UI o instala los modelos faltantes con `ollama pull`.")

        ok = len(models) > 0 and not missing_required
        return CheckResult(
            ok=ok,
            warnings=warnings,
            suggestions=suggestions,
            details={"detected_models": models, "missing_required_models": missing_required},
        )

    def validate_disk_access(self) -> CheckResult:
        """Valida acceso de escritura para carpetas de datos, índice y logs."""
        errors: List[str] = []
        writable_paths: List[str] = []
        for path in self.storage_paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path, prefix=".env_check_", delete=True) as tmpf:
                    tmpf.write(b"ok")
                    tmpf.flush()
                writable_paths.append(str(path))
            except Exception as exc:
                errors.append(f"No hay acceso de escritura en {path}: {exc}")

        if errors:
            return CheckResult(
                ok=False,
                errors=errors,
                suggestions=["Revisa permisos del directorio del proyecto o ejecuta con un usuario con acceso de escritura."],
                details={"writable_paths": writable_paths},
            )

        return CheckResult(ok=True, details={"writable_paths": writable_paths})

    def validate(self) -> Dict[str, Any]:
        """Ejecuta validación completa y retorna payload serializable para API/UI."""
        checks = {
            "python": self.validate_python(),
            "dependencies": self.validate_dependencies(),
            "ollama": self.validate_ollama(),
            "disk": self.validate_disk_access(),
        }

        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        serialized_checks: Dict[str, Any] = {}
        for name, result in checks.items():
            serialized_checks[name] = {
                "ok": result.ok,
                "errors": result.errors,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
                "details": result.details,
            }
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            suggestions.extend(result.suggestions)

        return {
            "ok": all(r.ok for r in checks.values()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": serialized_checks,
            "errors": errors,
            "warnings": warnings,
            "suggestions": sorted(set(suggestions)),
        }


def get_environment_status() -> Dict[str, Any]:
    """Helper para obtener estado de entorno en una llamada."""
    return EnvironmentValidator().validate()


def main() -> int:
    """CLI amigable para preflight manual."""
    status = get_environment_status()
    print("== Environment Validator ==")
    print(f"Overall OK: {status['ok']}")
    for section, payload in status["checks"].items():
        icon = "OK" if payload["ok"] else "WARN"
        print(f"[{icon}] {section}")
        for key in ("errors", "warnings", "suggestions"):
            for item in payload.get(key, []):
                print(f"  - {item}")
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
