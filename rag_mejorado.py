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
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

os.environ.setdefault("USER_AGENT", "Manobot/1.0")

WIKI_URL = os.getenv("RAG_SOURCE_URL", "https://es.wikipedia.org/wiki/Inteligencia_artificial")
DATA_FOLDER = Path(os.getenv("RAG_DATA_DIR", "./data"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "llama3.1:8b")
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", "./chroma_db"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "mi_rag_multifuente")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_CHARS_PER_CHUNK = int(os.getenv("RAG_MAX_CHARS_PER_CHUNK", "1200"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("RAG_MAX_TOTAL_CONTEXT_CHARS", "4200"))
ENABLE_CONTEXT_COMPRESSION = os.getenv("RAG_ENABLE_CONTEXT_COMPRESSION", "0") == "1"
SEMANTIC_CACHE_ENABLED = os.getenv("RAG_SEMANTIC_CACHE_ENABLED", "1") == "1"

REGISTRY_FILE = CHROMA_DIR / "index_registry.json"
SEMANTIC_CACHE_FILE = CHROMA_DIR / "semantic_cache.json"


# ---------------------------------------------------------------------------
# Logging JSON estructurado
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Formatter de logs en JSON para facilitar ingesta en sistemas observabilidad."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": int(record.created * 1000),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rag_mejorado")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _build_logger()


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
            cls._embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            logger.info("Embeddings inicializado")
        return cls._embeddings

    @classmethod
    def llm(cls) -> Any:
        if cls._llm is None:
            _, _, _, _, OllamaLLM = _lazy_import_langchain()
            cls._llm = OllamaLLM(model=CHAT_MODEL)
            logger.info("LLM inicializado")
        return cls._llm


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
    raw = f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{COLLECTION_NAME}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


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


def get_runtime_config() -> Dict[str, str | int | bool]:
    return {
        "wiki_url": WIKI_URL,
        "data_folder": str(DATA_FOLDER),
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
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
    }


def get_data_inventory() -> Dict[str, List[str] | int]:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p.name for p in DATA_FOLDER.glob("*.pdf"))
    txts = sorted(p.name for p in DATA_FOLDER.glob("*.txt"))
    return {
        "pdf_files": pdfs,
        "txt_files": txts,
        "pdf_count": len(pdfs),
        "txt_count": len(txts),
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

    try:
        loaded = WebBaseLoader(
            web_paths=(WIKI_URL,),
            bs_kwargs={"parse_only": bs4.SoupStrainer(id="bodyContent")},
        ).load()
        for d in loaded:
            d.metadata = _enrich_metadata(d, WIKI_URL, "web")
        docs.extend(loaded)
        stats.web_ok += 1
        logger.info(f"Web cargada: {WIKI_URL}")
    except Exception as exc:
        stats.web_errors += 1
        logger.error(f"Error web {WIKI_URL}: {exc}")

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

    current: Dict[str, Dict[str, str]] = {}
    changed_docs: List[Any] = []
    touched_source_ids: List[str] = []
    new_docs = 0
    updated_docs = 0

    for d in docs:
        meta = d.metadata
        sid = _source_id(meta["source"], meta["file_type"])
        h = meta["original_document_id"]
        current[sid] = {
            "source": meta["source"],
            "file_type": meta["file_type"],
            "hash": h,
            "embedding_version": emb_version,
        }

        old = reg_sources.get(sid)
        if old is None:
            new_docs += 1
            changed_docs.append(d)
            touched_source_ids.append(sid)
        elif old.get("hash") != h or old.get("embedding_version") != emb_version:
            updated_docs += 1
            changed_docs.append(d)
            touched_source_ids.append(sid)

    deleted_source_ids = [sid for sid in reg_sources if sid not in current]

    registry["embedding_version"] = emb_version
    registry["sources"] = current

    stats = {
        "new_docs": new_docs,
        "updated_docs": updated_docs,
        "deleted_docs": len(deleted_source_ids),
    }
    return changed_docs, touched_source_ids + deleted_source_ids, stats


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
            delete_ids_prefix = []
            for sid in touched_source_ids:
                # borrado best-effort por ids sintéticos previos (rango razonable)
                for idx in range(0, 5000):
                    delete_ids_prefix.append(f"{sid}:{idx}:{registry.get('embedding_version', 'old')}")
            if delete_ids_prefix:
                vectorstore.delete(ids=delete_ids_prefix)
        except Exception as exc:
            logger.warning(f"Delete incremental best-effort falló: {exc}")

    if changed_docs:
        chunks = _split_documents(changed_docs)
        ids = _chunk_ids(chunks)
        vectorstore.add_documents(chunks, ids=ids)
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
    top_k: int = TOP_K,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_mode: str = "similarity",
) -> Tuple[List[Any], float, float]:
    """Retrieval con filtro metadata y opción MMR."""
    start = _now_ms()
    docs: List[Any] = []
    avg_score = 0.0

    kwargs: Dict[str, Any] = {"k": top_k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter

    if search_mode == "mmr":
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=kwargs)
        docs = retriever.invoke(question)
        # score de referencia adicional para auditoría
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=top_k, filter=metadata_filter)
    else:
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=top_k, filter=metadata_filter)
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
        content = (getattr(d, "page_content", "") or "").strip()[:MAX_CHARS_PER_CHUNK]
        section = f"--- chunk {i} | source={source} | type={m.get('file_type')} | page={m.get('page_number')} ---\n{content}\n"
        if total + len(section) > MAX_TOTAL_CONTEXT_CHARS:
            break
        parts.append(section)
        total += len(section)

    return _compress_context_if_enabled("\n".join(parts).strip())


def _build_prompt(question: str, context: str) -> str:
    """Prompt robusto con formato JSON y anti-alucinación."""
    return f"""
Eres un asistente experto en análisis documental.
Reglas obligatorias:
1) NO alucines. Usa únicamente el contexto.
2) Si falta información suficiente, responde exactamente: "No lo sé".
3) Responde en español, mínimo 4 líneas.
4) Devuelve ÚNICAMENTE JSON válido con esta forma:
{{
  "answer": "respuesta en texto",
  "confidence": "baja|media|alta",
  "sources": [
    {{"source": "ruta_o_url", "file_type": "pdf|txt|web", "page_number": 0}}
  ]
}}

Contexto:
{context}

Pregunta: {question}
""".strip()


def rag_chat(
    question: str,
    vectorstore: Any,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_mode: str = "similarity",
) -> str:
    """Consulta RAG con cache semántico, retrieval avanzado y métricas."""
    if not question.strip():
        return "Por favor, envía una pregunta no vacía."

    cache = SemanticCache(SEMANTIC_CACHE_FILE)
    if SEMANTIC_CACHE_ENABLED:
        cached = cache.get(question)
        if cached:
            logger.info("Semantic cache hit")
            return cached

    metrics = Metrics()

    try:
        docs, avg_score, retrieval_ms = _retrieve(
            vectorstore=vectorstore,
            question=question,
            top_k=TOP_K,
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

    prompt = _build_prompt(question, context)

    gen_start = _now_ms()
    try:
        answer = ModelRegistry.llm().invoke(prompt)
        metrics.generation_latency_ms = round(_now_ms() - gen_start, 2)
    except Exception as exc:
        metrics.error_count += 1
        logger.error(f"Error generación: {exc}")
        return f"No pude generar respuesta: {exc}"

    logger.info(f"Métricas consulta: {asdict(metrics)}")

    if SEMANTIC_CACHE_ENABLED:
        cache.put(question, answer)
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
    print(rag_chat(pregunta, vectorstore))


if __name__ == "__main__":
    main()
