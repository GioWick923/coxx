"""Gestión de perfiles de modelos/configuración como módulo dedicado."""

from __future__ import annotations

from typing import Any, Dict, Optional


def list_model_profiles() -> Dict[str, Dict[str, Any]]:
    import rag_mejorado as rag

    return rag.list_model_profiles()


def upsert_model_profile(
    name: str,
    chat_model: str,
    embed_model: str,
    description: str = "",
    generation: Optional[Dict[str, Any]] = None,
    rag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.upsert_model_profile(
        name=name,
        chat_model=chat_model,
        embed_model=embed_model,
        description=description,
        generation=generation,
        rag=rag,
    )


def apply_model_profile(name: str) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.apply_model_profile(name)


def delete_model_profile(name: str) -> bool:
    import rag_mejorado as rag

    return rag.delete_model_profile(name)
