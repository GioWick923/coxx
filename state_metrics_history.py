"""Estado operacional, métricas, cache e historial del runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional


def get_model_metrics() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.get_model_metrics()


def get_cache_status() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.get_cache_status()


def clear_cache() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.clear_cache()


def get_history(query: Optional[str] = None, model: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.get_history(query=query, model=model, limit=limit)


def clear_history() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.clear_history()


def reload_config() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.reload_config()
