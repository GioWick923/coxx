"""Funciones núcleo de RAG extraídas como fachada estable."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def load_or_create_vectorstore(force_rebuild: bool = False) -> Tuple[Any, Dict[str, Any] | None]:
    """Carga/crea índice vectorial preservando comportamiento actual."""
    import rag_mejorado as rag

    return rag.load_or_create_vectorstore(force_rebuild=force_rebuild)


def rag_chat(
    question: str,
    vectorstore: Any,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_mode: str = "similarity",
) -> str:
    """Ejecuta consulta RAG con el pipeline existente."""
    import rag_mejorado as rag

    return rag.rag_chat(
        question=question,
        vectorstore=vectorstore,
        metadata_filter=metadata_filter,
        search_mode=search_mode,
    )


def set_active_models(chat_model: str | None = None, embed_model: str | None = None, auto_detect: bool = False):
    """Ajusta modelos activos manteniendo la lógica de validación vigente."""
    import rag_mejorado as rag

    return rag.set_active_models(chat_model=chat_model, embed_model=embed_model, auto_detect=auto_detect)


def get_runtime_config():
    """Retorna configuración runtime consolidada."""
    import rag_mejorado as rag

    return rag.get_runtime_config()


def get_data_inventory():
    """Retorna inventario de fuentes disponibles."""
    import rag_mejorado as rag

    return rag.get_data_inventory()


def add_web_source(url: str) -> str:
    """Agrega una fuente web al catálogo persistente."""
    import rag_mejorado as rag

    return rag.add_web_source(url)
