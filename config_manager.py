"""Gestión centralizada de configuración basada en .env."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict


class ConfigValidationError(ValueError):
    """Error de validación de configuración."""


@dataclass
class AppConfig:
    # Modelos
    rag_embed_model: str = "nomic-embed-text"
    rag_chat_model: str = "llama3.1:8b"

    # RAG
    rag_source_url: str = "https://es.wikipedia.org/wiki/Inteligencia_artificial"
    rag_data_dir: str = "./data"
    rag_chroma_dir: str = "./chroma_db"
    rag_collection: str = "mi_rag_multifuente"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 4
    rag_max_chars_per_chunk: int = 1200
    rag_max_total_context_chars: int = 4200

    # Flags
    rag_enable_context_compression: bool = False
    rag_semantic_cache_enabled: bool = True

    # Puertos
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860

    # Runtime / logging
    user_agent: str = "Manobot/1.0"
    rag_log_level: str = "INFO"
    rag_log_file: str = "./logs/rag_backend.log"
    rag_log_max_bytes: int = 1048576
    rag_log_backup_count: int = 5


class ConfigManager:
    """Carga, valida y expone configuración tipada desde `.env` + entorno."""

    def __init__(self, env_file: str | Path = ".env") -> None:
        self.env_file = Path(env_file)
        self._lock = Lock()
        self._config = AppConfig()
        self.reload()

    @staticmethod
    def _parse_bool(value: str | bool | None, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on", "y", "si", "sí"}:
            return True
        if raw in {"0", "false", "no", "off", "n"}:
            return False
        raise ConfigValidationError(f"Valor booleano inválido: {value}")

    @staticmethod
    def _parse_int(value: str | None, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except Exception as exc:
            raise ConfigValidationError(f"Valor entero inválido: {value}") from exc

    def _load_env_file(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        if not self.env_file.exists():
            return values
        for line in self.env_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            values[key] = val
        return values

    def _get(self, file_values: Dict[str, str], key: str, default: Any) -> Any:
        return os.getenv(key, file_values.get(key, default))

    def reload(self) -> AppConfig:
        with self._lock:
            file_values = self._load_env_file()
            cfg = AppConfig(
                rag_embed_model=str(self._get(file_values, "RAG_EMBED_MODEL", AppConfig.rag_embed_model)).strip(),
                rag_chat_model=str(self._get(file_values, "RAG_CHAT_MODEL", AppConfig.rag_chat_model)).strip(),
                rag_source_url=str(self._get(file_values, "RAG_SOURCE_URL", AppConfig.rag_source_url)).strip(),
                rag_data_dir=str(self._get(file_values, "RAG_DATA_DIR", AppConfig.rag_data_dir)).strip(),
                rag_chroma_dir=str(self._get(file_values, "RAG_CHROMA_DIR", AppConfig.rag_chroma_dir)).strip(),
                rag_collection=str(self._get(file_values, "RAG_COLLECTION", AppConfig.rag_collection)).strip(),
                rag_chunk_size=self._parse_int(self._get(file_values, "RAG_CHUNK_SIZE", None), AppConfig.rag_chunk_size),
                rag_chunk_overlap=self._parse_int(self._get(file_values, "RAG_CHUNK_OVERLAP", None), AppConfig.rag_chunk_overlap),
                rag_top_k=self._parse_int(self._get(file_values, "RAG_TOP_K", None), AppConfig.rag_top_k),
                rag_max_chars_per_chunk=self._parse_int(self._get(file_values, "RAG_MAX_CHARS_PER_CHUNK", None), AppConfig.rag_max_chars_per_chunk),
                rag_max_total_context_chars=self._parse_int(self._get(file_values, "RAG_MAX_TOTAL_CONTEXT_CHARS", None), AppConfig.rag_max_total_context_chars),
                rag_enable_context_compression=self._parse_bool(self._get(file_values, "RAG_ENABLE_CONTEXT_COMPRESSION", None), AppConfig.rag_enable_context_compression),
                rag_semantic_cache_enabled=self._parse_bool(self._get(file_values, "RAG_SEMANTIC_CACHE_ENABLED", None), AppConfig.rag_semantic_cache_enabled),
                ui_host=str(self._get(file_values, "UI_HOST", AppConfig.ui_host)).strip(),
                ui_port=self._parse_int(self._get(file_values, "UI_PORT", None), AppConfig.ui_port),
                user_agent=str(self._get(file_values, "USER_AGENT", AppConfig.user_agent)).strip(),
                rag_log_level=str(self._get(file_values, "RAG_LOG_LEVEL", AppConfig.rag_log_level)).strip().upper(),
                rag_log_file=str(self._get(file_values, "RAG_LOG_FILE", AppConfig.rag_log_file)).strip(),
                rag_log_max_bytes=self._parse_int(self._get(file_values, "RAG_LOG_MAX_BYTES", None), AppConfig.rag_log_max_bytes),
                rag_log_backup_count=self._parse_int(self._get(file_values, "RAG_LOG_BACKUP_COUNT", None), AppConfig.rag_log_backup_count),
            )
            self._validate(cfg)
            self._config = cfg
            return cfg

    @staticmethod
    def _validate(cfg: AppConfig) -> None:
        if cfg.rag_chunk_size <= 0:
            raise ConfigValidationError("RAG_CHUNK_SIZE debe ser > 0")
        if cfg.rag_chunk_overlap < 0:
            raise ConfigValidationError("RAG_CHUNK_OVERLAP no puede ser negativo")
        if cfg.rag_top_k <= 0:
            raise ConfigValidationError("RAG_TOP_K debe ser > 0")
        if cfg.ui_port <= 0 or cfg.ui_port > 65535:
            raise ConfigValidationError("UI_PORT fuera de rango")
        if cfg.rag_chunk_overlap >= cfg.rag_chunk_size:
            raise ConfigValidationError("RAG_CHUNK_OVERLAP debe ser menor que RAG_CHUNK_SIZE")
        if cfg.rag_log_max_bytes <= 0:
            raise ConfigValidationError("RAG_LOG_MAX_BYTES debe ser > 0")
        if cfg.rag_log_backup_count < 1:
            raise ConfigValidationError("RAG_LOG_BACKUP_COUNT debe ser >= 1")

    def get(self) -> AppConfig:
        with self._lock:
            return self._config

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self.get())


CONFIG = ConfigManager()
