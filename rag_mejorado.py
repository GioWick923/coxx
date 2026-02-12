"""RAG avanzado con indexado incremental, metadata enriquecida y auditoría.

Este módulo mantiene compatibilidad con la UI local (`web_ui.py`) a través de:
- `get_runtime_config()`
- `get_data_inventory()`
- `load_or_create_vectorstore(force_rebuild=False)`
- `rag_chat(question, vectorstore, ...)`

Diseño:
- Imports pesados/librerías externas se cargan de forma lazy para que el módulo sea importable
  incluso si faltan dependencias (útil para auditoría, tests y mensajes de estado).
- Logging estructurado JSON para producción.
- Metadata enriquecida en documento y chunks (source, file_type, page_number,
  original_document_id/hash, embedding_version).
- Indexado incremental con registro local (hash/timestamp por fuente).
- Retrieval con filtros metadata y modo MMR.
- Cache semántico opcional para acelerar preguntas repetidas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import threading
import platform
import uuid
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config_manager import CONFIG
from structured_logging import get_structured_logger

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

_CFG = CONFIG.get()
os.environ.setdefault("USER_AGENT", _CFG.user_agent)
os.environ.setdefault("RAG_LOG_LEVEL", _CFG.rag_log_level)
os.environ.setdefault("RAG_LOG_FILE", _CFG.rag_log_file)
os.environ.setdefault("RAG_LOG_MAX_BYTES", str(_CFG.rag_log_max_bytes))
os.environ.setdefault("RAG_LOG_BACKUP_COUNT", str(_CFG.rag_log_backup_count))

WIKI_URL = _CFG.rag_source_url
DATA_FOLDER = Path(_CFG.rag_data_dir)
EMBED_MODEL = _CFG.rag_embed_model
CHAT_MODEL = _CFG.rag_chat_model
CHROMA_DIR = Path(_CFG.rag_chroma_dir)
COLLECTION_NAME = _CFG.rag_collection

CHUNK_SIZE = _CFG.rag_chunk_size
CHUNK_OVERLAP = _CFG.rag_chunk_overlap
TOP_K = _CFG.rag_top_k
MAX_CHARS_PER_CHUNK = _CFG.rag_max_chars_per_chunk
MAX_TOTAL_CONTEXT_CHARS = _CFG.rag_max_total_context_chars
ENABLE_CONTEXT_COMPRESSION = _CFG.rag_enable_context_compression
SEMANTIC_CACHE_ENABLED = _CFG.rag_semantic_cache_enabled
SMART_CACHE_TTL_SECONDS = int(os.getenv("RAG_SMART_CACHE_TTL_SECONDS", "21600"))
SMART_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SMART_CACHE_SIMILARITY_THRESHOLD", "0.82"))

REGISTRY_FILE = CHROMA_DIR / "index_registry.json"
SEMANTIC_CACHE_FILE = CHROMA_DIR / "semantic_cache.json"
WEB_SOURCES_FILE = DATA_FOLDER / "web_sources.txt"
MODEL_PROFILES_FILE = CHROMA_DIR / "model_profiles.json"
HISTORY_FILE = CHROMA_DIR / "query_history.json"
BENCHMARK_RESULTS_FILE = CHROMA_DIR / "model_benchmark_results.json"
BACKUP_DIR = CHROMA_DIR / "backups"
BACKUP_SCHEMA_VERSION = 1
BACKUP_INTERVAL_SECONDS = int(os.getenv("RAG_BACKUP_INTERVAL_SECONDS", "3600"))


ACTIVE_EMBED_MODEL = EMBED_MODEL
ACTIVE_CHAT_MODEL = CHAT_MODEL
ACTIVE_GENERATION_CONFIG: Dict[str, float] = {"temperature": 0.1, "top_p": 0.9}
ACTIVE_RAG_CONFIG: Dict[str, int] = {
    "top_k": TOP_K,
    "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
    "max_total_context_chars": MAX_TOTAL_CONTEXT_CHARS,
}
_MODEL_LOCK = threading.Lock()



logger = get_structured_logger("rag_mejorado")


@dataclass
class ModelPerfStats:
    """Métricas acumuladas por modelo/tarea en tiempo real."""

    request_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.request_count, 2)


class ModelMetricsTracker:
    """Tracker en memoria para latencias y uso de memoria del proceso.

    Formato snapshot (JSON-ready):
    {
      "active_models": {"chat": "llama3.1:8b", "embedding": "nomic-embed-text"},
      "detected_models": [{"name": "llama3.1:8b", "version": "8b", "type": "chat"}],
      "performance": {
        "chat": {"model": "llama3.1:8b", "request_count": 3, "avg_latency_ms": 120.3},
        "embedding": {"model": "nomic-embed-text", "request_count": 2, "avg_latency_ms": 44.8}
      },
      "memory": {"rss_mb": 123.4, "source": "resource/psutil/unavailable"}
    }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, ModelPerfStats]] = {
            "chat": {},
            "embedding": {},
            "deep_analysis": {},
        }

    @staticmethod
    def _detect_memory_mb() -> Tuple[Optional[float], str]:
        # Prioridad: resource (unix) -> psutil (si existe) -> unavailable.
        try:
            import resource  # type: ignore

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux: KB, macOS: bytes
            if platform.system().lower() == "darwin":
                return round(rss_kb / (1024 * 1024), 2), "resource"
            return round(rss_kb / 1024, 2), "resource"
        except Exception:
            pass

        try:
            import psutil  # type: ignore

            rss = psutil.Process(os.getpid()).memory_info().rss
            return round(rss / (1024 * 1024), 2), "psutil"
        except Exception:
            return None, "unavailable"

    @staticmethod
    def _model_version(model_name: str) -> str:
        if ":" in model_name:
            return model_name.split(":", 1)[1]
        return "latest"

    def record(self, task: str, model_name: str, latency_ms: float) -> None:
        task_key = normalize_task(task)
        with self._lock:
            bucket = self._stats.setdefault(task_key, {})
            stat = bucket.setdefault(model_name, ModelPerfStats())
            stat.request_count += 1
            stat.total_latency_ms += float(latency_ms)

    def snapshot(self, detected_models: Optional[List[str]] = None) -> Dict[str, Any]:
        models = detected_models if detected_models is not None else list_ollama_models()
        rss_mb, source = self._detect_memory_mb()

        def _best(task: str, active_name: str) -> Dict[str, Any]:
            task_bucket = self._stats.get(task, {})
            active_stat = task_bucket.get(active_name)
            if active_stat is not None:
                return {
                    "model": active_name,
                    "request_count": active_stat.request_count,
                    "avg_latency_ms": active_stat.avg_latency_ms,
                }
            return {"model": active_name, "request_count": 0, "avg_latency_ms": 0.0}

        with self._lock:
            perf = {
                "chat": _best("chat", ACTIVE_CHAT_MODEL),
                "embedding": _best("embedding", ACTIVE_EMBED_MODEL),
                "deep_analysis": _best("deep_analysis", ACTIVE_CHAT_MODEL),
            }

        detected = [
            {
                "name": name,
                "version": self._model_version(name),
                "type": classify_model_type(name),
            }
            for name in models
        ]

        return {
            "active_models": {
                "chat": {"name": ACTIVE_CHAT_MODEL, "version": self._model_version(ACTIVE_CHAT_MODEL)},
                "embedding": {"name": ACTIVE_EMBED_MODEL, "version": self._model_version(ACTIVE_EMBED_MODEL)},
            },
            "detected_models": detected,
            "performance": perf,
            "memory": {"rss_mb": rss_mb, "source": source},
        }


MODEL_METRICS = ModelMetricsTracker()


DEFAULT_MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "Rapido": {
        "description": "Prioriza velocidad de respuesta con modelos livianos.",
        "chat_model": "llama3.1:8b",
        "embed_model": "nomic-embed-text",
        "generation": {"temperature": 0.2, "top_p": 0.85},
        "rag": {"top_k": 3, "max_chars_per_chunk": 900, "max_total_context_chars": 3200},
    },
    "Preciso": {
        "description": "Prioriza calidad/razonamiento con modelos más robustos.",
        "chat_model": "deepseek-r1:32b",
        "embed_model": "nomic-embed-text",
        "generation": {"temperature": 0.05, "top_p": 0.95},
        "rag": {"top_k": 5, "max_chars_per_chunk": 1200, "max_total_context_chars": 4600},
    },
    "Bajo consumo": {
        "description": "Minimiza uso de recursos para equipos modestos.",
        "chat_model": "phi3:mini",
        "embed_model": "nomic-embed-text",
        "generation": {"temperature": 0.15, "top_p": 0.8},
        "rag": {"top_k": 2, "max_chars_per_chunk": 700, "max_total_context_chars": 2400},
    },
}


def _normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "description": str(raw.get("description", "Perfil personalizado de modelos Ollama.")).strip(),
        "chat_model": str(raw.get("chat_model", "")).strip(),
        "embed_model": str(raw.get("embed_model", "")).strip(),
        "generation": {
            "temperature": float(raw.get("generation", {}).get("temperature", ACTIVE_GENERATION_CONFIG["temperature"])),
            "top_p": float(raw.get("generation", {}).get("top_p", ACTIVE_GENERATION_CONFIG["top_p"])),
        },
        "rag": {
            "top_k": int(raw.get("rag", {}).get("top_k", ACTIVE_RAG_CONFIG["top_k"])),
            "max_chars_per_chunk": int(raw.get("rag", {}).get("max_chars_per_chunk", ACTIVE_RAG_CONFIG["max_chars_per_chunk"])),
            "max_total_context_chars": int(raw.get("rag", {}).get("max_total_context_chars", ACTIVE_RAG_CONFIG["max_total_context_chars"])),
        },
    }


def _validate_profile(name: str, profile: Dict[str, Any]) -> None:
    if not name.strip():
        raise ValueError("El nombre del perfil no puede estar vacío")
    if not profile.get("chat_model") or not profile.get("embed_model"):
        raise ValueError("chat_model y embed_model son obligatorios")


def _load_model_profiles() -> Dict[str, Dict[str, Any]]:
    """Carga perfiles desde JSON; si no existe, usa defaults."""
    if MODEL_PROFILES_FILE.exists():
        try:
            data = json.loads(MODEL_PROFILES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): _normalize_profile(v or {}) for k, v in data.items()}
        except Exception:
            logger.warning("Archivo de perfiles inválido, se usarán defaults")
    return {k: _normalize_profile(v) for k, v in DEFAULT_MODEL_PROFILES.items()}


def _save_model_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PROFILES_FILE.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def list_model_profiles() -> Dict[str, Dict[str, Any]]:
    """Retorna todos los perfiles de modelos disponibles."""
    profiles = _load_model_profiles()
    if not MODEL_PROFILES_FILE.exists():
        _save_model_profiles(profiles)
    return profiles


def upsert_model_profile(
    name: str,
    chat_model: str,
    embed_model: str,
    description: str = "",
    generation: Optional[Dict[str, Any]] = None,
    rag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Crea o edita un perfil de configuración completa (modelos + generación + RAG)."""
    profile_name = name.strip()
    candidate = _normalize_profile(
        {
            "description": description,
            "chat_model": chat_model,
            "embed_model": embed_model,
            "generation": generation or {},
            "rag": rag or {},
        }
    )
    _validate_profile(profile_name, candidate)

    profiles = _load_model_profiles()
    profiles[profile_name] = candidate
    _save_model_profiles(profiles)
    return {"name": profile_name, **candidate}


def delete_model_profile(name: str) -> bool:
    """Elimina un perfil existente."""
    profile_name = name.strip()
    profiles = _load_model_profiles()
    if profile_name not in profiles:
        raise ValueError(f"Perfil no encontrado: {profile_name}")
    del profiles[profile_name]
    _save_model_profiles(profiles)
    return True


def apply_model_profile(name: str) -> Dict[str, Any]:
    """Aplica un perfil guardado actualizando modelos activos."""
    profiles = _load_model_profiles()
    profile_name = name.strip()
    if profile_name not in profiles:
        raise ValueError(f"Perfil no encontrado: {profile_name}")

    selected = profiles[profile_name]
    result = set_active_models(
        chat_model=selected["chat_model"],
        embed_model=selected["embed_model"],
        auto_detect=False,
    )

    with _MODEL_LOCK:
        ACTIVE_GENERATION_CONFIG.update(selected.get("generation", {}))
        ACTIVE_RAG_CONFIG.update(selected.get("rag", {}))
        ModelRegistry.reset()

    return {
        "profile": profile_name,
        "description": selected.get("description", ""),
        "chat_model": result["chat_model"],
        "embed_model": result["embed_model"],
        "generation": dict(ACTIVE_GENERATION_CONFIG),
        "rag": dict(ACTIVE_RAG_CONFIG),
        "installed_models": result.get("installed_models", []),
    }


CONFIG_EXPORT_VERSION = 1

DEFAULT_BENCHMARK_PROMPTS = [
    "Explica brevemente qué es inteligencia artificial y menciona dos riesgos.",
    "Resume en 3 puntos prácticos cómo usar RAG para responder preguntas sobre documentos.",
]


def _load_benchmark_results() -> List[Dict[str, Any]]:
    if BENCHMARK_RESULTS_FILE.exists():
        try:
            data = json.loads(BENCHMARK_RESULTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            logger.warning("Archivo benchmark inválido, se reinicia histórico")
    return []


def _save_benchmark_results(items: List[Dict[str, Any]]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _estimate_answer_quality(answer: str, prompt: str) -> Dict[str, Any]:
    """Heurística simple de calidad para benchmark automático.

    Puntúa por:
    - longitud mínima útil,
    - cobertura de palabras clave del prompt,
    - estructura con viñetas/numeración (para respuestas accionables).
    """
    text = (answer or "").strip()
    prompt_tokens = {t for t in prompt.lower().replace(".", " ").split() if len(t) > 4}
    answer_tokens = set(text.lower().replace(".", " ").split())
    overlap = len(prompt_tokens & answer_tokens)
    coverage = overlap / max(1, len(prompt_tokens))
    length_score = min(len(text) / 500.0, 1.0)
    structure_score = 1.0 if any(tok in text for tok in ("- ", "1)", "1.", "•")) else 0.5
    quality = round(min(1.0, (0.45 * coverage) + (0.35 * length_score) + (0.20 * structure_score)), 3)
    return {
        "score": quality,
        "coverage": round(coverage, 3),
        "length_chars": len(text),
        "structure_score": structure_score,
    }


def run_model_benchmark(
    prompts: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    limit_models: int = 5,
) -> Dict[str, Any]:
    """Ejecuta benchmark automático comparando modelos de chat detectados.

    Métricas por modelo:
    - `avg_latency_ms`
    - `avg_quality_score` (heurística estimada)
    - `memory_rss_mb` (estimación del proceso al ejecutar)
    """
    prompts_to_use = [p.strip() for p in (prompts or DEFAULT_BENCHMARK_PROMPTS) if p and p.strip()]
    if not prompts_to_use:
        raise ValueError("Se requiere al menos un prompt para benchmark")

    installed = list_ollama_models()
    grouped = classify_models(installed)
    candidate_models = models if models is not None else grouped.get(TASK_CHAT, [])
    candidate_models = [m for m in candidate_models if m in installed][: max(1, min(limit_models, 10))]
    if not candidate_models:
        raise ValueError("No hay modelos de chat disponibles para benchmark")

    _, _, _, _, OllamaLLM = _lazy_import_langchain()

    bench_items: List[Dict[str, Any]] = []
    started_ms = int(time.time() * 1000)
    for model_name in candidate_models:
        latencies: List[float] = []
        quality_scores: List[float] = []
        prompt_results: List[Dict[str, Any]] = []
        for prompt in prompts_to_use:
            t0 = _now_ms()
            try:
                llm = OllamaLLM(model=model_name, **ACTIVE_GENERATION_CONFIG)
                answer = llm.invoke(prompt)
                latency = round(_now_ms() - t0, 2)
                latencies.append(latency)
                q = _estimate_answer_quality(str(answer), prompt)
                quality_scores.append(q["score"])
                prompt_results.append(
                    {
                        "prompt": prompt,
                        "latency_ms": latency,
                        "quality": q,
                    }
                )
            except Exception as exc:
                prompt_results.append(
                    {
                        "prompt": prompt,
                        "error": str(exc),
                    }
                )

        rss_mb, mem_source = MODEL_METRICS._detect_memory_mb()
        if rss_mb is None:
            rss_mb = 0.0
        bench_items.append(
            {
                "model": model_name,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
                "memory_rss_mb": rss_mb,
                "memory_source": mem_source,
                "prompt_results": prompt_results,
                "ok_prompts": len(latencies),
                "failed_prompts": len([r for r in prompt_results if "error" in r]),
            }
        )

    run = {
        "run_id": int(time.time() * 1000),
        "started_at_ms": started_ms,
        "finished_at_ms": int(time.time() * 1000),
        "prompts": prompts_to_use,
        "models_tested": candidate_models,
        "results": bench_items,
    }
    history = _load_benchmark_results()
    history.append(run)
    history = history[-200:]
    _save_benchmark_results(history)
    return run


def get_benchmark_results(limit: int = 20) -> Dict[str, Any]:
    """Devuelve histórico de benchmark persistente.

    Ejemplo JSON:
    {
      "count": 1,
      "results": [
        {
          "run_id": 1730000000000,
          "results": [
            {
              "model": "llama3.1:8b",
              "avg_latency_ms": 523.3,
              "avg_quality_score": 0.74,
              "memory_rss_mb": 311.2
            }
          ]
        }
      ]
    }
    """
    all_items = list(reversed(_load_benchmark_results()))
    safe_limit = max(1, min(int(limit or 20), 100))
    selected = all_items[:safe_limit]
    return {
        "count": len(selected),
        "results": selected,
        "benchmark_file": str(BENCHMARK_RESULTS_FILE),
    }


def _validate_models_exist(models: List[str], installed_models: Optional[List[str]] = None) -> None:
    """Valida que los modelos solicitados existan en Ollama."""
    installed = installed_models if installed_models is not None else list_ollama_models()
    if not installed:
        raise ValueError("No se pudieron validar modelos: Ollama no disponible o sin modelos instalados")
    missing = sorted({m for m in models if m and m not in installed})
    if missing:
        raise ValueError(f"Modelos no encontrados en Ollama local: {', '.join(missing)}")


def export_configuration() -> Dict[str, Any]:
    """Exporta configuración global completa en JSON serializable."""
    return {
        "schema_version": CONFIG_EXPORT_VERSION,
        "exported_at_ms": int(time.time() * 1000),
        "profiles": list_model_profiles(),
        "active": {
            "chat_model": ACTIVE_CHAT_MODEL,
            "embed_model": ACTIVE_EMBED_MODEL,
            "generation": dict(ACTIVE_GENERATION_CONFIG),
            "rag": dict(ACTIVE_RAG_CONFIG),
        },
        "runtime": get_runtime_config(),
    }


def import_configuration(payload: Dict[str, Any], overwrite_profiles: bool = False) -> Dict[str, Any]:
    """Importa configuración global con validaciones y manejo de conflictos.

    - No sobrescribe perfiles existentes por defecto.
    - Valida integridad de esquema y existencia de modelos en Ollama.
    """
    if not isinstance(payload, dict):
        raise ValueError("El payload de importación debe ser un objeto JSON")

    required = {"schema_version", "profiles", "active"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Archivo inválido. Faltan campos: {', '.join(missing)}")

    if int(payload.get("schema_version", 0)) != CONFIG_EXPORT_VERSION:
        raise ValueError("Versión de esquema no compatible")

    profiles_in = payload.get("profiles")
    if not isinstance(profiles_in, dict):
        raise ValueError("Campo 'profiles' inválido")

    active_in = payload.get("active")
    if not isinstance(active_in, dict):
        raise ValueError("Campo 'active' inválido")

    # Normaliza y valida perfiles
    normalized_profiles: Dict[str, Dict[str, Any]] = {}
    models_to_validate: List[str] = []
    for name, raw_profile in profiles_in.items():
        profile_name = str(name).strip()
        norm = _normalize_profile(raw_profile or {})
        _validate_profile(profile_name, norm)
        normalized_profiles[profile_name] = norm
        models_to_validate.extend([norm["chat_model"], norm["embed_model"]])

    active_chat = str(active_in.get("chat_model", "")).strip()
    active_embed = str(active_in.get("embed_model", "")).strip()
    if not active_chat or not active_embed:
        raise ValueError("Active config inválida: chat_model/embed_model requeridos")

    active_gen = active_in.get("generation") if isinstance(active_in.get("generation"), dict) else {}
    active_rag = active_in.get("rag") if isinstance(active_in.get("rag"), dict) else {}

    models_to_validate.extend([active_chat, active_embed])
    _validate_models_exist(models_to_validate)

    existing = list_model_profiles()
    conflicts = sorted(set(existing.keys()) & set(normalized_profiles.keys()))
    if conflicts and not overwrite_profiles:
        raise ValueError(
            "Conflicto de perfiles existentes. Usa overwrite_profiles=true para sobrescribir: "
            + ", ".join(conflicts)
        )

    merged = dict(existing)
    merged.update(normalized_profiles)
    _save_model_profiles(merged)

    # Aplica configuración activa importada
    set_active_models(chat_model=active_chat, embed_model=active_embed, auto_detect=False)
    normalized_active = _normalize_profile(
        {
            "chat_model": active_chat,
            "embed_model": active_embed,
            "generation": active_gen,
            "rag": active_rag,
        }
    )
    with _MODEL_LOCK:
        ACTIVE_GENERATION_CONFIG.update(normalized_active["generation"])
        ACTIVE_RAG_CONFIG.update(normalized_active["rag"])
        ModelRegistry.reset()

    return {
        "imported_profiles": len(normalized_profiles),
        "overwritten_profiles": len(conflicts) if overwrite_profiles else 0,
        "applied_active": {
            "chat_model": ACTIVE_CHAT_MODEL,
            "embed_model": ACTIVE_EMBED_MODEL,
            "generation": dict(ACTIVE_GENERATION_CONFIG),
            "rag": dict(ACTIVE_RAG_CONFIG),
        },
    }




def _safe_load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return default


def run_backup(reason: str = "manual") -> Dict[str, Any]:
    """Crea backup versionado de perfiles, configuración, historial y métricas."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    backup_id = f"backup_{ts_ms}"
    backup_file = BACKUP_DIR / f"{backup_id}.json"

    payload = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "backup_id": backup_id,
        "created_at_ms": ts_ms,
        "reason": reason,
        "profiles": list_model_profiles(),
        "configuration": export_configuration(),
        "history": _safe_load_json_file(HISTORY_FILE, []),
        "metrics": {
            "model_metrics": get_model_metrics(),
            "benchmark_results": _load_benchmark_results(),
        },
    }

    backup_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "backup_id": backup_id,
        "backup_file": str(backup_file),
        "reason": reason,
        "summary": {
            "profiles": len(payload["profiles"]),
            "history_items": len(payload["history"]) if isinstance(payload["history"], list) else 0,
            "benchmark_runs": len(payload["metrics"].get("benchmark_results", [])),
        },
    }


def list_backups(limit: int = 50) -> Dict[str, Any]:
    """Lista backups disponibles ordenados del más reciente al más antiguo."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    safe_limit = max(1, min(int(limit or 50), 200))
    items = []
    for file_path in sorted(BACKUP_DIR.glob("backup_*.json"), reverse=True):
        stat = file_path.stat()
        items.append(
            {
                "backup_id": file_path.stem,
                "file": str(file_path),
                "size_bytes": stat.st_size,
                "modified_at_ms": int(stat.st_mtime * 1000),
            }
        )
    selected = items[:safe_limit]
    return {"count": len(selected), "items": selected, "backup_dir": str(BACKUP_DIR)}


def restore_backup(backup_id: str, overwrite_profiles: bool = False) -> Dict[str, Any]:
    """Restaura backup versionado con validación básica de integridad."""
    normalized = str(backup_id or "").strip()
    if not normalized:
        raise ValueError("backup_id es requerido")

    backup_file = BACKUP_DIR / f"{normalized}.json"
    if not backup_file.exists():
        raise ValueError("Backup no encontrado")

    payload = _safe_load_json_file(backup_file, None)
    if not isinstance(payload, dict):
        raise ValueError("Archivo de backup inválido")

    if int(payload.get("schema_version", 0)) != BACKUP_SCHEMA_VERSION:
        raise ValueError("Versión de backup no compatible")

    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Backup inválido: falta bloque de configuración")

    import_result = import_configuration(configuration, overwrite_profiles=overwrite_profiles)

    history_rows = payload.get("history")
    if isinstance(history_rows, list):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        global HISTORY_STORE
        HISTORY_STORE = QueryHistoryStore(HISTORY_FILE)

    metrics_block = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    benchmark_rows = metrics_block.get("benchmark_results")
    if isinstance(benchmark_rows, list):
        _save_benchmark_results(benchmark_rows)

    return {
        "ok": True,
        "backup_id": normalized,
        "restored": {
            "profiles_imported": import_result.get("imported_profiles", 0),
            "history_items": len(history_rows) if isinstance(history_rows, list) else 0,
            "benchmark_runs": len(benchmark_rows) if isinstance(benchmark_rows, list) else 0,
        },
    }


class BackupScheduler:
    """Scheduler simple para backups automáticos periódicos."""

    def __init__(self, interval_seconds: int) -> None:
        self.interval_seconds = max(60, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="backup-scheduler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                run_backup(reason="scheduled")
            except Exception as exc:
                logger.warning(f"Backup automático falló: {exc}")

    def stop(self) -> None:
        self._stop_event.set()


_BACKUP_SCHEDULER: Optional[BackupScheduler] = None


def start_backup_scheduler(interval_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Inicia scheduler de backup automático si está habilitado."""
    global _BACKUP_SCHEDULER
    interval = int(interval_seconds if interval_seconds is not None else BACKUP_INTERVAL_SECONDS)
    if interval <= 0:
        return {"ok": True, "started": False, "reason": "disabled", "interval_seconds": interval}

    if _BACKUP_SCHEDULER is None:
        _BACKUP_SCHEDULER = BackupScheduler(interval)
    _BACKUP_SCHEDULER.start()
    return {"ok": True, "started": True, "interval_seconds": _BACKUP_SCHEDULER.interval_seconds}


def reload_config() -> Dict[str, Any]:
    """Recarga configuración desde .env y aplica cambios runtime seguros."""
    global _CFG, WIKI_URL, DATA_FOLDER, EMBED_MODEL, CHAT_MODEL, CHROMA_DIR
    global COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
    global MAX_CHARS_PER_CHUNK, MAX_TOTAL_CONTEXT_CHARS, ENABLE_CONTEXT_COMPRESSION
    global SEMANTIC_CACHE_ENABLED, REGISTRY_FILE, SEMANTIC_CACHE_FILE, WEB_SOURCES_FILE
    global MODEL_PROFILES_FILE, HISTORY_FILE, BENCHMARK_RESULTS_FILE, BACKUP_DIR, BACKUP_INTERVAL_SECONDS

    _CFG = CONFIG.reload()
    WIKI_URL = _CFG.rag_source_url
    DATA_FOLDER = Path(_CFG.rag_data_dir)
    EMBED_MODEL = _CFG.rag_embed_model
    CHAT_MODEL = _CFG.rag_chat_model
    CHROMA_DIR = Path(_CFG.rag_chroma_dir)
    COLLECTION_NAME = _CFG.rag_collection
    CHUNK_SIZE = _CFG.rag_chunk_size
    CHUNK_OVERLAP = _CFG.rag_chunk_overlap
    TOP_K = _CFG.rag_top_k
    MAX_CHARS_PER_CHUNK = _CFG.rag_max_chars_per_chunk
    MAX_TOTAL_CONTEXT_CHARS = _CFG.rag_max_total_context_chars
    ENABLE_CONTEXT_COMPRESSION = _CFG.rag_enable_context_compression
    SEMANTIC_CACHE_ENABLED = _CFG.rag_semantic_cache_enabled

    REGISTRY_FILE = CHROMA_DIR / "index_registry.json"
    SEMANTIC_CACHE_FILE = CHROMA_DIR / "semantic_cache.json"
    WEB_SOURCES_FILE = DATA_FOLDER / "web_sources.txt"
    MODEL_PROFILES_FILE = CHROMA_DIR / "model_profiles.json"
    HISTORY_FILE = CHROMA_DIR / "query_history.json"
    BENCHMARK_RESULTS_FILE = CHROMA_DIR / "model_benchmark_results.json"
    BACKUP_DIR = CHROMA_DIR / "backups"
    BACKUP_INTERVAL_SECONDS = int(os.getenv("RAG_BACKUP_INTERVAL_SECONDS", "3600"))

    with _MODEL_LOCK:
        ModelRegistry.reset()

    return {"ok": True, "config": CONFIG.as_dict()}


# ---------------------------------------------------------------------------
# Excepciones y métricas
# ---------------------------------------------------------------------------


class RAGSetupError(RuntimeError):
    """Error de inicialización/configuración del pipeline RAG."""


@dataclass
class LoadStats:
    pdf_ok: int = 0
    txt_ok: int = 0
    web_ok: int = 0
    pdf_errors: int = 0
    txt_errors: int = 0
    web_errors: int = 0
    total_docs: int = 0


@dataclass
class Metrics:
    index_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    chunks_by_type: Dict[str, int] = field(default_factory=dict)
    average_retrieval_score: float = 0.0
    fallback_count: int = 0
    error_count: int = 0
    indexed_new_docs: int = 0
    indexed_updated_docs: int = 0
    indexed_deleted_docs: int = 0
    split_chunks: int = 0


# ---------------------------------------------------------------------------
# Model manager (reuse de instancias)
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Singleton simple para reutilizar embeddings/LLM y evitar overhead."""

    _embeddings: Any = None
    _llm: Any = None

    @classmethod
    def embeddings(cls) -> Any:
        if cls._embeddings is None:
            _, _, _, OllamaEmbeddings, _ = _lazy_import_langchain()
            cls._embeddings = OllamaEmbeddings(model=ACTIVE_EMBED_MODEL)
            logger.info("Embeddings inicializado")
        return cls._embeddings

    @classmethod
    def llm(cls) -> Any:
        if cls._llm is None:
            _, _, _, _, OllamaLLM = _lazy_import_langchain()
            cls._llm = OllamaLLM(model=ACTIVE_CHAT_MODEL, **ACTIVE_GENERATION_CONFIG)
            logger.info("LLM inicializado")
        return cls._llm

    @classmethod
    def reset(cls) -> None:
        """Reinicia instancias para aplicar cambio de modelo en caliente."""
        cls._embeddings = None
        cls._llm = None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


class SemanticCache:
    """Cache semántico básico por normalización de pregunta y hash."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    @staticmethod
    def _key(question: str) -> str:
        normalized = " ".join(question.lower().strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[str]:
        return self._data.get(self._key(question))

    def put(self, question: str, answer: str) -> None:
        self._data[self._key(question)] = answer
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")


class SmartResponseCache:
    """Cache inteligente con similitud de consultas, TTL y limpieza manual.

    Cada entrada guarda:
    - question
    - context (texto usado en prompt)
    - answer
    - model
    - created_at / expires_at
    """

    def __init__(self, path: Path, ttl_seconds: int = SMART_CACHE_TTL_SECONDS):
        self.path = path
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "entries": [],
        }
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._state["hits"] = int(loaded.get("hits", 0))
                    self._state["misses"] = int(loaded.get("misses", 0))
                    self._state["entries"] = list(loaded.get("entries", []))
            except Exception:
                logger.warning("No fue posible leer smart cache existente, se reinicia")

    @staticmethod
    def _normalize(text: str) -> str:
        base = (text or "").lower().strip()
        base = "".join(
            ch for ch in unicodedata.normalize("NFD", base) if unicodedata.category(ch) != "Mn"
        )
        return " ".join(base.split())

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = SmartResponseCache._normalize(text)
        return {t for t in re.findall(r"[\wáéíóúñü]+", normalized, flags=re.IGNORECASE) if t}

    @staticmethod
    def _context_fingerprint(context: str) -> str:
        return hashlib.sha256((context or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        a_tokens = SmartResponseCache._tokenize(a)
        b_tokens = SmartResponseCache._tokenize(b)
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return inter / union if union else 0.0

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _prune_expired(self) -> None:
        now = time.time()
        self._state["entries"] = [
            e for e in self._state.get("entries", []) if float(e.get("expires_at", 0)) > now
        ]

    def get(self, question: str, context: str, model: str) -> Optional[str]:
        """Busca match exacto/similar para la misma huella de contexto y modelo."""
        with self._lock:
            self._prune_expired()
            normalized = self._normalize(question)
            ctx_fp = self._context_fingerprint(context)

            best_entry: Optional[Dict[str, Any]] = None
            best_score = 0.0
            for entry in self._state.get("entries", []):
                if entry.get("model") != model:
                    continue
                if entry.get("context_fingerprint") != ctx_fp:
                    continue
                score = self._similarity(normalized, entry.get("normalized_question", ""))
                if score > best_score:
                    best_entry = entry
                    best_score = score

            if best_entry and best_score >= SMART_CACHE_SIMILARITY_THRESHOLD:
                self._state["hits"] = int(self._state.get("hits", 0)) + 1
                self._persist()
                return str(best_entry.get("answer", ""))

            self._state["misses"] = int(self._state.get("misses", 0)) + 1
            self._persist()
            return None

    def put(self, question: str, context: str, answer: str, model: str) -> None:
        with self._lock:
            self._prune_expired()
            created = time.time()
            entry = {
                "question": question,
                "normalized_question": self._normalize(question),
                "context": context,
                "context_fingerprint": self._context_fingerprint(context),
                "answer": answer,
                "model": model,
                "created_at": created,
                "expires_at": created + self.ttl_seconds,
            }
            self._state["entries"].append(entry)
            # Tope defensivo para no crecer indefinidamente.
            self._state["entries"] = self._state["entries"][-500:]
            self._persist()

    def clear(self) -> Dict[str, Any]:
        with self._lock:
            removed = len(self._state.get("entries", []))
            self._state["entries"] = []
            self._persist()
            return {"removed_entries": removed}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_expired()
            entries = self._state.get("entries", [])
            return {
                "enabled": SEMANTIC_CACHE_ENABLED,
                "ttl_seconds": self.ttl_seconds,
                "similarity_threshold": SMART_CACHE_SIMILARITY_THRESHOLD,
                "entries": len(entries),
                "hits": int(self._state.get("hits", 0)),
                "misses": int(self._state.get("misses", 0)),
                "cache_file": str(self.path),
            }


SMART_CACHE = SmartResponseCache(SEMANTIC_CACHE_FILE)


def get_cache_status() -> Dict[str, Any]:
    """Expone estado del cache inteligente para API/UI."""
    return SMART_CACHE.status()


def clear_cache() -> Dict[str, Any]:
    """Limpieza manual del cache inteligente."""
    result = SMART_CACHE.clear()
    return {"ok": True, **result, **SMART_CACHE.status()}


class QueryHistoryStore:
    """Historial persistente de consultas con filtros simples para análisis posterior."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self._items = loaded
            except Exception:
                logger.warning("No fue posible leer historial existente, se reinicia")

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(
        self,
        *,
        question: str,
        model: str,
        latency_ms: float,
        documents: List[Dict[str, Any]],
        answer: str,
    ) -> Dict[str, Any]:
        with self._lock:
            item = {
                "id": str(uuid.uuid4()),
                "timestamp": int(time.time()),
                "question": question,
                "model": model,
                "latency_ms": round(float(latency_ms), 2),
                "documents": documents,
                "answer": answer,
            }
            self._items.append(item)
            self._items = self._items[-2000:]
            self._persist()
            return item

    def list(
        self,
        *,
        query: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(reversed(self._items))
        q = (query or "").strip().lower()
        m = (model or "").strip().lower()
        if q:
            rows = [
                r
                for r in rows
                if q in str(r.get("question", "")).lower() or q in str(r.get("answer", "")).lower()
            ]
        if m:
            rows = [r for r in rows if str(r.get("model", "")).lower() == m]
        safe_limit = max(1, min(int(limit or 100), 500))
        return rows[:safe_limit]

    def clear(self) -> Dict[str, Any]:
        with self._lock:
            removed = len(self._items)
            self._items = []
            self._persist()
            return {"removed": removed}


HISTORY_STORE = QueryHistoryStore(HISTORY_FILE)


def get_history(
    query: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    items = HISTORY_STORE.list(query=query, model=model, limit=limit)
    return {
        "items": items,
        "count": len(items),
        "filters": {"query": query or "", "model": model or "", "limit": max(1, min(int(limit or 100), 500))},
        "history_file": str(HISTORY_FILE),
    }


def clear_history() -> Dict[str, Any]:
    result = HISTORY_STORE.clear()
    return {"ok": True, **result, "history_file": str(HISTORY_FILE)}


def _lazy_import_langchain() -> Tuple[Any, Any, Any, Any, Any]:
    """Importa dependencias LangChain de forma lazy para evitar fallos al importar módulo."""
    try:
        import bs4
        from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
    except Exception as exc:
        raise RAGSetupError(f"Dependencias de RAG no disponibles: {exc}") from exc
    return bs4, PyPDFLoader, TextLoader, OllamaEmbeddings, OllamaLLM


def _lazy_import_chroma_and_splitter() -> Tuple[Any, Any, Any]:
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise RAGSetupError(f"Dependencias de vectorstore/splitter no disponibles: {exc}") from exc
    return Chroma, Document, RecursiveCharacterTextSplitter


def _doc_hash(content: str, source: str, file_type: str) -> str:
    raw = f"{source}|{file_type}|{content}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _embedding_version() -> str:
    """Versionado de embeddings por modelo + chunking + colección."""
    raw = f"{ACTIVE_EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{COLLECTION_NAME}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def list_ollama_models() -> List[str]:
    """Lista modelos disponibles en Ollama local usando `ollama list`."""
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception as exc:
        logger.warning(f"No fue posible consultar modelos Ollama: {exc}")
        return []

    if proc.returncode != 0:
        logger.warning(f"`ollama list` devolvió código {proc.returncode}: {proc.stderr.strip()}")
        return []

    models: List[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        model = line.split()[0]
        if model and model not in models:
            models.append(model)
    return models


TASK_CHAT = "chat"
TASK_EMBEDDING = "embedding"
TASK_DEEP_ANALYSIS = "deep_analysis"


# Heurísticas simples y explícitas por tipo de modelo/tarea.
EMBED_KEYWORDS = ("embed", "nomic-embed", "bge", "e5", "mxbai")
DEEP_ANALYSIS_KEYWORDS = ("70b", "34b", "32b", "27b", "qwen", "deepseek", "coder", "r1")
CHAT_KEYWORDS = ("llama", "mistral", "gemma", "qwen", "phi")


def classify_model_type(model_name: str) -> str:
    """Clasifica un modelo Ollama en: embedding, deep_analysis o chat.

    Reglas:
    - Si contiene tokens de embeddings (embed, bge, e5, ...): embedding.
    - Si no es embedding y parece modelo grande/analítico: deep_analysis.
    - En otro caso: chat.
    """
    name = (model_name or "").lower()
    if any(k in name for k in EMBED_KEYWORDS):
        return TASK_EMBEDDING
    if any(k in name for k in DEEP_ANALYSIS_KEYWORDS):
        return TASK_DEEP_ANALYSIS
    return TASK_CHAT


def classify_models(installed_models: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Agrupa modelos instalados por tipo de tarea.

    Ejemplo:
        input: ["nomic-embed-text", "llama3.1:8b", "deepseek-r1:32b"]
        output: {
            "embedding": ["nomic-embed-text"],
            "chat": ["llama3.1:8b"],
            "deep_analysis": ["deepseek-r1:32b"],
        }
    """
    models = installed_models if installed_models is not None else list_ollama_models()
    grouped: Dict[str, List[str]] = {
        TASK_EMBEDDING: [],
        TASK_CHAT: [],
        TASK_DEEP_ANALYSIS: [],
    }
    for model in models:
        grouped[classify_model_type(model)].append(model)
    return grouped


def select_best_model_for_task(
    task: str,
    installed_models: Optional[List[str]] = None,
    fallback: Optional[str] = None,
) -> str:
    """Selecciona el mejor modelo según tipo de tarea solicitada.

    Tareas soportadas: chat, embedding, deep_analysis.
    """
    grouped = classify_models(installed_models)
    all_models = installed_models if installed_models is not None else list_ollama_models()

    if task == TASK_EMBEDDING and grouped[TASK_EMBEDDING]:
        return grouped[TASK_EMBEDDING][0]

    if task == TASK_DEEP_ANALYSIS:
        if grouped[TASK_DEEP_ANALYSIS]:
            return grouped[TASK_DEEP_ANALYSIS][0]
        # fallback fuerte: chat más "grande" encontrado.
        if grouped[TASK_CHAT]:
            return sorted(grouped[TASK_CHAT], key=lambda m: len(m), reverse=True)[0]

    if task == TASK_CHAT and grouped[TASK_CHAT]:
        return grouped[TASK_CHAT][0]

    if all_models:
        return all_models[0]

    if fallback:
        return fallback

    raise RAGSetupError("No hay modelos disponibles para seleccionar")


def suggest_models(installed_models: Optional[List[str]] = None) -> Dict[str, str]:
    """Sugiere modelos base (embeddings/chat/deep_analysis) en base a Ollama local."""
    models = installed_models if installed_models is not None else list_ollama_models()
    if not models:
        return {
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "analysis_model": CHAT_MODEL,
        }

    return {
        "embed_model": select_best_model_for_task(TASK_EMBEDDING, models, fallback=EMBED_MODEL),
        "chat_model": select_best_model_for_task(TASK_CHAT, models, fallback=CHAT_MODEL),
        "analysis_model": select_best_model_for_task(TASK_DEEP_ANALYSIS, models, fallback=CHAT_MODEL),
    }


def normalize_task(task: Optional[str]) -> str:
    """Normaliza alias de tareas para selección de modelo."""
    t = (task or "").strip().lower()
    aliases = {
        "chat": TASK_CHAT,
        "conversation": TASK_CHAT,
        "conversacional": TASK_CHAT,
        "embedding": TASK_EMBEDDING,
        "embeddings": TASK_EMBEDDING,
        "vector": TASK_EMBEDDING,
        "deep": TASK_DEEP_ANALYSIS,
        "analysis": TASK_DEEP_ANALYSIS,
        "analisis": TASK_DEEP_ANALYSIS,
        "análisis": TASK_DEEP_ANALYSIS,
        "deep_analysis": TASK_DEEP_ANALYSIS,
    }
    return aliases.get(t, TASK_CHAT)


def select_models_for_tasks(
    chat_task: str = TASK_CHAT,
    embedding_task: str = TASK_EMBEDDING,
    analysis_task: str = TASK_DEEP_ANALYSIS,
    installed_models: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Selecciona modelos para las tres tareas principales.

    Esto sirve como API compacta para UI/backend.
    """
    models = installed_models if installed_models is not None else list_ollama_models()
    return {
        "chat_model": select_best_model_for_task(normalize_task(chat_task), models, fallback=CHAT_MODEL),
        "embed_model": select_best_model_for_task(normalize_task(embedding_task), models, fallback=EMBED_MODEL),
        "analysis_model": select_best_model_for_task(normalize_task(analysis_task), models, fallback=CHAT_MODEL),
    }


def set_active_models(
    chat_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    auto_detect: bool = False,
) -> Dict[str, str | List[str]]:
    """Configura modelos activos manual o automáticamente sin reiniciar la app.

    Regla de seguridad:
    - Si Ollama reporta modelos instalados y se recibe uno manual que no existe,
      se lanza ValueError para fallo controlado.
    """
    global ACTIVE_CHAT_MODEL, ACTIVE_EMBED_MODEL
    installed = list_ollama_models()
    if auto_detect:
        suggested = suggest_models(installed)
        chat_model = suggested["chat_model"]
        embed_model = suggested["embed_model"]

    normalized_chat = chat_model.strip() if chat_model else None
    normalized_embed = embed_model.strip() if embed_model else None

    if installed:
        if normalized_chat and normalized_chat not in installed:
            raise ValueError(f"Modelo chat no encontrado en Ollama local: {normalized_chat}")
        if normalized_embed and normalized_embed not in installed:
            raise ValueError(f"Modelo embedding no encontrado en Ollama local: {normalized_embed}")

    with _MODEL_LOCK:
        if normalized_chat:
            ACTIVE_CHAT_MODEL = normalized_chat
        if normalized_embed:
            ACTIVE_EMBED_MODEL = normalized_embed
        ModelRegistry.reset()

    return {
        "chat_model": ACTIVE_CHAT_MODEL,
        "embed_model": ACTIVE_EMBED_MODEL,
        "analysis_model": suggest_models(installed)["analysis_model"] if installed else ACTIVE_CHAT_MODEL,
        "installed_models": installed,
        "classified_models": classify_models(installed),
    }


def _load_registry() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Registro de indexado inválido; se recreará")
    return {"embedding_version": _embedding_version(), "sources": {}}


def _save_registry(registry: Dict[str, Any]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_id(source: str, file_type: str) -> str:
    return hashlib.sha256(f"{file_type}:{source}".encode("utf-8")).hexdigest()[:18]


def _now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------------------------------------------------------------------
# Inventario/config pública
# ---------------------------------------------------------------------------




def list_web_sources() -> List[str]:
    """Retorna fuentes web configuradas (base + adicionales en data/web_sources.txt)."""
    urls = [WIKI_URL]
    if WEB_SOURCES_FILE.exists():
        for line in WEB_SOURCES_FILE.read_text(encoding="utf-8").splitlines():
            url = line.strip()
            if url.startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
    return urls


def add_web_source(url: str) -> str:
    """Agrega una URL al catálogo local de fuentes web."""
    clean = url.strip()
    if not clean.startswith(("http://", "https://")):
        raise ValueError("La URL debe iniciar con http:// o https://")
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    urls = list_web_sources()
    if clean in urls:
        return clean
    WEB_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with WEB_SOURCES_FILE.open("a", encoding="utf-8") as f:
        if WEB_SOURCES_FILE.stat().st_size > 0:
            f.write("\n")
        f.write(clean)
    return clean

def get_model_metrics() -> Dict[str, Any]:
    """Retorna métricas runtime de modelos activos/detectados en formato JSON-serializable.

    Ejemplo de salida esperada:
    {
      "active_models": {
        "chat": {"name": "llama3.1:8b", "version": "8b"},
        "embedding": {"name": "nomic-embed-text", "version": "latest"}
      },
      "detected_models": [
        {"name": "llama3.1:8b", "version": "8b", "type": "chat"}
      ],
      "performance": {
        "chat": {"model": "llama3.1:8b", "request_count": 4, "avg_latency_ms": 131.2},
        "embedding": {"model": "nomic-embed-text", "request_count": 1, "avg_latency_ms": 45.8},
        "deep_analysis": {"model": "llama3.1:8b", "request_count": 0, "avg_latency_ms": 0.0}
      },
      "memory": {"rss_mb": 205.8, "source": "resource"}
    }
    """
    return MODEL_METRICS.snapshot(list_ollama_models())


def get_runtime_config() -> Dict[str, str | int | bool]:
    installed = list_ollama_models()
    return {
        "wiki_url": WIKI_URL,
        "data_folder": str(DATA_FOLDER),
        "embed_model_default": EMBED_MODEL,
        "chat_model_default": CHAT_MODEL,
        "embed_model": ACTIVE_EMBED_MODEL,
        "chat_model": ACTIVE_CHAT_MODEL,
        "installed_ollama_models": installed,
        "classified_ollama_models": classify_models(installed),
        "task_model_selection": select_models_for_tasks(installed_models=installed),
        "chroma_dir": str(CHROMA_DIR),
        "collection_name": COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
        "max_total_context_chars": MAX_TOTAL_CONTEXT_CHARS,
        "embedding_version": _embedding_version(),
        "context_compression": ENABLE_CONTEXT_COMPRESSION,
        "semantic_cache": SEMANTIC_CACHE_ENABLED,
        "web_sources_file": str(WEB_SOURCES_FILE),
        "model_profiles_file": str(MODEL_PROFILES_FILE),
        "backup_dir": str(BACKUP_DIR),
        "backup_interval_seconds": BACKUP_INTERVAL_SECONDS,
        "generation_config": dict(ACTIVE_GENERATION_CONFIG),
        "rag_runtime_config": dict(ACTIVE_RAG_CONFIG),
    }


def get_data_inventory() -> Dict[str, List[str] | int]:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p.name for p in DATA_FOLDER.glob("*.pdf"))
    txts = sorted(p.name for p in DATA_FOLDER.glob("*.txt"))
    web_urls = list_web_sources()
    return {
        "pdf_files": pdfs,
        "txt_files": txts,
        "pdf_count": len(pdfs),
        "txt_count": len(txts),
        "web_urls": web_urls,
        "web_count": len(web_urls),
    }


# ---------------------------------------------------------------------------
# Ingesta con metadata enriquecida
# ---------------------------------------------------------------------------


def _enrich_metadata(document: Any, source: str, file_type: str) -> Dict[str, Any]:
    """Construye metadata base por documento para trazabilidad y auditoría."""
    meta = dict(getattr(document, "metadata", {}) or {})
    content = getattr(document, "page_content", "")
    page_number = meta.get("page")
    meta.update(
        {
            "source": source,
            "source_id": _source_id(source, file_type),
            "file_type": file_type,
            "page_number": int(page_number) if isinstance(page_number, int) else None,
            "original_document_id": _doc_hash(content, source, file_type),
            "embedding_version": _embedding_version(),
        }
    )
    return meta


def _load_documents() -> Tuple[List[Any], LoadStats]:
    """Carga documentos de PDF/TXT/Web con metadata enriquecida."""
    bs4, PyPDFLoader, TextLoader, _, _ = _lazy_import_langchain()
    _, Document, _ = _lazy_import_chroma_and_splitter()

    docs: List[Any] = []
    stats = LoadStats()
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    for pdf in DATA_FOLDER.glob("*.pdf"):
        try:
            loaded = PyPDFLoader(str(pdf)).load()
            for d in loaded:
                d.metadata = _enrich_metadata(d, str(pdf), "pdf")
            docs.extend(loaded)
            stats.pdf_ok += 1
            logger.info(f"PDF cargado: {pdf}")
        except Exception as exc:
            stats.pdf_errors += 1
            logger.error(f"Error PDF {pdf}: {exc}")

    for txt in DATA_FOLDER.glob("*.txt"):
        try:
            loaded = TextLoader(str(txt), encoding="utf-8").load()
            for d in loaded:
                d.metadata = _enrich_metadata(d, str(txt), "txt")
            docs.extend(loaded)
            stats.txt_ok += 1
            logger.info(f"TXT cargado: {txt}")
        except Exception as exc:
            stats.txt_errors += 1
            logger.error(f"Error TXT {txt}: {exc}")

    for web_url in list_web_sources():
        try:
            loaded = WebBaseLoader(
                web_paths=(web_url,),
                bs_kwargs={"parse_only": bs4.SoupStrainer(id="bodyContent")},
            ).load()
            for d in loaded:
                d.metadata = _enrich_metadata(d, web_url, "web")
            docs.extend(loaded)
            stats.web_ok += 1
            logger.info(f"Web cargada: {web_url}")
        except Exception as exc:
            stats.web_errors += 1
            logger.error(f"Error web {web_url}: {exc}")

    stats.total_docs = len(docs)
    if not docs:
        raise RAGSetupError("No se cargó ningún documento.")

    # Coerción a Document explícito (compatibilidad)
    normalized: List[Any] = []
    for d in docs:
        normalized.append(Document(page_content=getattr(d, "page_content", ""), metadata=d.metadata))
    return normalized, stats


# ---------------------------------------------------------------------------
# Indexado incremental/versionado
# ---------------------------------------------------------------------------


def _split_documents(docs: List[Any]) -> List[Any]:
    _, _, RecursiveCharacterTextSplitter = _lazy_import_chroma_and_splitter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for idx, ch in enumerate(chunks):
        ch.metadata = dict(ch.metadata)
        ch.metadata["chunk_index"] = idx
    return chunks


def _classify_changes(docs: List[Any], registry: Dict[str, Any]) -> Tuple[List[Any], List[str], Dict[str, int]]:
    """Determina documentos nuevos/actualizados/borrados usando hash y versionado."""
    emb_version = _embedding_version()
    reg_sources: Dict[str, Any] = dict(registry.get("sources", {}))

    # Calculamos hash por fuente (archivo/url) en vez de por página/chunk para
    # que el indexado incremental sea estable y no marque cambios falsos.
    grouped: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        meta = d.metadata
        sid = meta.get("source_id") or _source_id(meta["source"], meta["file_type"])
        entry = grouped.setdefault(
            sid,
            {
                "source": meta["source"],
                "file_type": meta["file_type"],
                "doc_hashes": [],
            },
        )
        entry["doc_hashes"].append(meta["original_document_id"])

    current: Dict[str, Dict[str, str]] = {}
    changed_source_ids: set[str] = set()
    for sid, data in grouped.items():
        joined = "|".join(sorted(data["doc_hashes"]))
        source_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        current[sid] = {
            "source": data["source"],
            "file_type": data["file_type"],
            "hash": source_hash,
            "embedding_version": emb_version,
        }

        old = reg_sources.get(sid)
        if old is None or old.get("hash") != source_hash or old.get("embedding_version") != emb_version:
            changed_source_ids.add(sid)

    new_docs = len([sid for sid in changed_source_ids if sid not in reg_sources])
    updated_docs = len([sid for sid in changed_source_ids if sid in reg_sources])
    changed_docs = [d for d in docs if d.metadata.get("source_id") in changed_source_ids]

    deleted_source_ids = [sid for sid in reg_sources if sid not in current]

    registry["embedding_version"] = emb_version
    registry["sources"] = current

    stats = {
        "new_docs": new_docs,
        "updated_docs": updated_docs,
        "deleted_docs": len(deleted_source_ids),
    }
    return changed_docs, sorted(changed_source_ids | set(deleted_source_ids)), stats


def _chunk_ids(chunks: List[Any]) -> List[str]:
    ids = []
    for ch in chunks:
        m = ch.metadata
        sid = _source_id(m["source"], m["file_type"])
        ids.append(f"{sid}:{m.get('chunk_index', 0)}:{m.get('embedding_version', 'v0')}")
    return ids


def _ensure_vectorstore() -> Any:
    Chroma, _, _ = _lazy_import_chroma_and_splitter()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=ModelRegistry.embeddings(),
        collection_name=COLLECTION_NAME,
    )


def load_or_create_vectorstore(force_rebuild: bool = False) -> Tuple[Any, Dict[str, Any] | None]:
    """Carga/crea índice. Soporta actualización incremental por cambios de fuentes.

    Retorna (vectorstore, metrics_dict).
    """
    start = _now_ms()
    metrics = Metrics()

    if force_rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = _ensure_vectorstore()

    docs, load_stats = _load_documents()
    registry = _load_registry()

    changed_docs, touched_source_ids, change_stats = _classify_changes(docs, registry)

    # Borrar chunks previos de fuentes tocadas (actualizadas/borradas)
    if touched_source_ids:
        try:
            for sid in touched_source_ids:
                vectorstore.delete(where={"source_id": sid})
        except Exception as exc:
            logger.warning(f"Delete incremental best-effort falló: {exc}")

    if changed_docs:
        chunks = _split_documents(changed_docs)
        ids = _chunk_ids(chunks)
        embed_start = _now_ms()
        vectorstore.add_documents(chunks, ids=ids)
        MODEL_METRICS.record(TASK_EMBEDDING, ACTIVE_EMBED_MODEL, _now_ms() - embed_start)
        metrics.split_chunks = len(chunks)
        by_type: Dict[str, int] = {}
        for ch in chunks:
            ft = ch.metadata.get("file_type", "unknown")
            by_type[ft] = by_type.get(ft, 0) + 1
        metrics.chunks_by_type = by_type

    metrics.indexed_new_docs = change_stats["new_docs"]
    metrics.indexed_updated_docs = change_stats["updated_docs"]
    metrics.indexed_deleted_docs = change_stats["deleted_docs"]
    metrics.index_latency_ms = round(_now_ms() - start, 2)

    _save_registry(registry)

    result = {
        "load_stats": asdict(load_stats),
        "metrics": asdict(metrics),
    }
    logger.info(f"Indexado completado: {result}")
    return vectorstore, result


# ---------------------------------------------------------------------------
# Retrieval avanzado + context builder + generación
# ---------------------------------------------------------------------------


def _retrieve(
    vectorstore: Any,
    question: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_mode: str = "similarity",
) -> Tuple[List[Any], float, float]:
    """Retrieval con filtro metadata y opción MMR."""
    start = _now_ms()
    docs: List[Any] = []
    avg_score = 0.0

    k_value = int(top_k or ACTIVE_RAG_CONFIG.get("top_k", TOP_K))
    kwargs: Dict[str, Any] = {"k": k_value}
    if metadata_filter:
        kwargs["filter"] = metadata_filter

    if search_mode == "mmr":
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=kwargs)
        docs = retriever.invoke(question)
        # score de referencia adicional para auditoría
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=k_value, filter=metadata_filter)
    else:
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=k_value, filter=metadata_filter)
        docs = [d for d, _ in scored]

    if scored:
        avg_score = sum(score for _, score in scored) / len(scored)

    return docs, float(avg_score), round(_now_ms() - start, 2)


def _dedupe_docs(docs: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").strip()
        h = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        result.append(d)
    return result


def _compress_context_if_enabled(context: str) -> str:
    if not ENABLE_CONTEXT_COMPRESSION:
        return context
    # Compresión ligera local (no dependiente de LLM) para fallback estable.
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    return "\n".join(lines[:25])


def _build_context(docs: List[Any]) -> str:
    """Context builder inteligente: dedupe + agrupado por source + límites."""
    docs = _dedupe_docs(docs)
    parts: List[str] = []
    total = 0

    for i, d in enumerate(docs, 1):
        m = getattr(d, "metadata", {}) or {}
        source = m.get("source", "desconocido")
        max_chars = int(ACTIVE_RAG_CONFIG.get("max_chars_per_chunk", MAX_CHARS_PER_CHUNK))
        content = (getattr(d, "page_content", "") or "").strip()[:max_chars]
        section = f"--- chunk {i} | source={source} | type={m.get('file_type')} | page={m.get('page_number')} ---\n{content}\n"
        max_total = int(ACTIVE_RAG_CONFIG.get("max_total_context_chars", MAX_TOTAL_CONTEXT_CHARS))
        if total + len(section) > max_total:
            break
        parts.append(section)
        total += len(section)

    return _compress_context_if_enabled("\n".join(parts).strip())


PROMPT_SUMMARY_LONG_TEXT = """
Eres un asistente experto en síntesis documental.
Objetivo: resumir texto largo de forma fiel y útil.
Reglas:
1) NO alucines: usa solo el contexto.
2) Si falta información, di "No lo sé".
3) Entrega JSON válido con:
{
  "summary": "resumen ejecutivo",
  "key_points": ["punto 1", "punto 2", "punto 3"],
  "risks_or_limits": ["..."],
  "sources": [{"source": "ruta_o_url", "file_type": "pdf|txt|web", "page_number": 0}]
}
Contexto:
{context}
Instrucción de resumen: {question}
""".strip()

PROMPT_QA_SHORT_CONTEXT = """
Eres un asistente experto en QA con contexto corto.
Reglas:
1) Responde breve y precisa (3-6 líneas).
2) NO alucines; usa solo el contexto.
3) Si no hay datos suficientes: "No lo sé".
4) Devuelve JSON válido:
{
  "answer": "respuesta corta",
  "confidence": "baja|media|alta",
  "sources": [{"source": "ruta_o_url", "file_type": "pdf|txt|web", "page_number": 0}]
}
Contexto:
{context}
Pregunta: {question}
""".strip()

PROMPT_QA_LONG_CONTEXT = """
Eres un asistente experto en QA con múltiples fuentes y contexto largo.
Reglas:
1) Sintetiza información cruzada entre fragmentos.
2) No inventes: usa solo el contexto.
3) Si falta evidencia, responde "No lo sé".
4) Devuelve JSON válido:
{
  "answer": "respuesta estructurada",
  "evidence": ["hecho 1", "hecho 2"],
  "confidence": "baja|media|alta",
  "sources": [{"source": "ruta_o_url", "file_type": "pdf|txt|web", "page_number": 0}]
}
Contexto:
{context}
Pregunta: {question}
""".strip()

PROMPT_REASONING_STEPS = """
Eres un asistente experto en análisis paso a paso controlado.
Reglas:
1) Usa solo el contexto, sin alucinar.
2) Expón razonamiento útil de forma concisa y verificable.
3) Si no alcanza la evidencia: "No lo sé".
4) Devuelve JSON válido:
{
  "answer": "conclusión final",
  "steps": ["paso 1", "paso 2", "paso 3"],
  "confidence": "baja|media|alta",
  "sources": [{"source": "ruta_o_url", "file_type": "pdf|txt|web", "page_number": 0}]
}
Contexto:
{context}
Pregunta: {question}
""".strip()


def choose_prompt_template(question: str, context: str) -> str:
    """Elige plantilla adaptativa según tipo de consulta y longitud del contexto.

    Reglas de selección:
    - Si la pregunta sugiere resumen -> plantilla de resumen largo.
    - Si pide razonamiento paso a paso -> plantilla reasoning.
    - Si es QA normal, usa short vs long según tamaño de contexto.
    """
    q = (question or "").lower()
    context_len = len(context or "")

    summary_tokens = ("resume", "resumen", "sintetiza", "síntesis", "sumariza")
    reasoning_tokens = ("paso a paso", "razona", "explica cómo", "analiza", "comparar")

    if any(t in q for t in summary_tokens):
        return "summary_long"
    if any(t in q for t in reasoning_tokens):
        return "reasoning_steps"
    if context_len > 2200:
        return "qa_long"
    return "qa_short"


def _build_prompt(question: str, context: str) -> str:
    """Construye prompt adaptativo para resumen, QA corto/largo y reasoning."""
    template_key = choose_prompt_template(question, context)
    templates = {
        "summary_long": PROMPT_SUMMARY_LONG_TEXT,
        "qa_short": PROMPT_QA_SHORT_CONTEXT,
        "qa_long": PROMPT_QA_LONG_CONTEXT,
        "reasoning_steps": PROMPT_REASONING_STEPS,
    }
    template = templates[template_key]
    return template.replace("{question}", question).replace("{context}", context)


def rag_chat(
    question: str,
    vectorstore: Any,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_mode: str = "similarity",
) -> str:
    """Consulta RAG con cache semántico, retrieval avanzado y métricas."""
    if not question.strip():
        return "Por favor, envía una pregunta no vacía."

    metrics = Metrics()

    try:
        docs, avg_score, retrieval_ms = _retrieve(
            vectorstore=vectorstore,
            question=question,
            top_k=ACTIVE_RAG_CONFIG.get("top_k", TOP_K),
            metadata_filter=metadata_filter,
            search_mode=search_mode,
        )
        metrics.average_retrieval_score = round(avg_score, 4)
        metrics.retrieval_latency_ms = retrieval_ms
    except Exception as exc:
        metrics.error_count += 1
        logger.error(f"Error retrieval: {exc}")
        return f"No pude consultar la base vectorial: {exc}"

    context = _build_context(docs)
    if not context:
        metrics.fallback_count += 1
        logger.warning("Sin contexto relevante")
        return "No encontré contexto relevante para responder con confianza."

    if SEMANTIC_CACHE_ENABLED:
        cached = SMART_CACHE.get(question=question, context=context, model=ACTIVE_CHAT_MODEL)
        if cached:
            logger.info("Smart cache hit")
            return cached

    prompt = _build_prompt(question, context)

    gen_start = _now_ms()
    try:
        answer = ModelRegistry.llm().invoke(prompt)
        metrics.generation_latency_ms = round(_now_ms() - gen_start, 2)
        MODEL_METRICS.record(TASK_CHAT, ACTIVE_CHAT_MODEL, metrics.generation_latency_ms)
    except Exception as exc:
        metrics.error_count += 1
        logger.error(f"Error generación: {exc}")
        return f"No pude generar respuesta: {exc}"

    logger.info(f"Métricas consulta: {asdict(metrics)}")

    history_docs = [
        {
            "source": d.metadata.get("source", ""),
            "file_type": d.metadata.get("file_type", ""),
            "page_number": d.metadata.get("page_number"),
            "score": d.metadata.get("retrieval_score"),
        }
        for d in docs
    ]
    HISTORY_STORE.add(
        question=question,
        model=ACTIVE_CHAT_MODEL,
        latency_ms=metrics.generation_latency_ms,
        documents=history_docs,
        answer=answer,
    )

    if SEMANTIC_CACHE_ENABLED:
        SMART_CACHE.put(question=question, context=context, answer=answer, model=ACTIVE_CHAT_MODEL)
    return answer


def main() -> None:
    try:
        vectorstore, info = load_or_create_vectorstore(force_rebuild=False)
        logger.info(f"Inventario: {get_data_inventory()}")
        logger.info(f"Index info: {info}")
    except RAGSetupError as exc:
        logger.error(f"Error init RAG: {exc}")
        return

    pregunta = "¿Cuáles son los riesgos de la inteligencia artificial?"
    logger.info("Demo response", extra={"response": rag_chat(pregunta, vectorstore)})


if __name__ == "__main__":
    main()
