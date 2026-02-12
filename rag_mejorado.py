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
import subprocess
import time
import threading
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
WEB_SOURCES_FILE = DATA_FOLDER / "web_sources.txt"

ACTIVE_EMBED_MODEL = EMBED_MODEL
ACTIVE_CHAT_MODEL = CHAT_MODEL
_MODEL_LOCK = threading.Lock()


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
            cls._embeddings = OllamaEmbeddings(model=ACTIVE_EMBED_MODEL)
            logger.info("Embeddings inicializado")
        return cls._embeddings

    @classmethod
    def llm(cls) -> Any:
        if cls._llm is None:
            _, _, _, _, OllamaLLM = _lazy_import_langchain()
            cls._llm = OllamaLLM(model=ACTIVE_CHAT_MODEL)
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
    """Configura modelos activos manual o automáticamente sin reiniciar la app."""
    global ACTIVE_CHAT_MODEL, ACTIVE_EMBED_MODEL
    installed = list_ollama_models()
    if auto_detect:
        suggested = suggest_models(installed)
        chat_model = suggested["chat_model"]
        embed_model = suggested["embed_model"]

    with _MODEL_LOCK:
        if chat_model:
            ACTIVE_CHAT_MODEL = chat_model.strip()
        if embed_model:
            ACTIVE_EMBED_MODEL = embed_model.strip()
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
