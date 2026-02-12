"""Interfaz web local para el RAG (sin dependencias web extra)."""

from __future__ import annotations

import json
import os
import mimetypes
import threading
from email.parser import BytesParser
from email.policy import default as email_default_policy
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from config_manager import CONFIG
from environment_validator import get_environment_status
from structured_logging import get_structured_logger

_CFG = CONFIG.get()
os.environ.setdefault("RAG_LOG_LEVEL", _CFG.rag_log_level)
os.environ.setdefault("RAG_LOG_FILE", _CFG.rag_log_file)
os.environ.setdefault("RAG_LOG_MAX_BYTES", str(_CFG.rag_log_max_bytes))
os.environ.setdefault("RAG_LOG_BACKUP_COUNT", str(_CFG.rag_log_backup_count))
HOST = _CFG.ui_host
PORT = _CFG.ui_port
MAX_REQUEST_BYTES = 20_000
logger = get_structured_logger("web_ui")


def parse_multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    """Parsea multipart/form-data sin usar `cgi` (compatible con Python 3.13+).

    Se soporta un campo de archivo llamado `file` (contrato actual del endpoint).
    """
    if "multipart/form-data" not in (content_type or "").lower():
        raise ValueError("Se esperaba multipart/form-data")

    # Construimos un mensaje MIME sintético para reutilizar el parser estándar `email`.
    synthetic = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    msg = BytesParser(policy=email_default_policy).parsebytes(synthetic)

    if not msg.is_multipart():
        raise ValueError("Payload multipart inválido")

    for part in msg.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue

        filename = part.get_filename()
        if not filename:
            raise ValueError("Archivo vacío")
        payload = part.get_payload(decode=True) or b""
        return filename, payload

    raise ValueError("No se encontró el campo file")


class AppState:
    def __init__(self) -> None:
        self.vectorstore = None
        self.metrics = None
        self.last_error = None
        self.environment_status = None
        self.lock = threading.Lock()


STATE = AppState()

INDEX_HTML_PATH = Path("ui_assets/index.html")
DEFAULT_INDEX_HTML = """<!doctype html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>RAG Studio</title></head><body style="background:#120a04;color:#ffd36b;font-family:Consolas,monospace;padding:16px"><h2>RAG Studio</h2><p>La plantilla principal no se pudo cargar; se muestra interfaz de respaldo.</p><div style="margin:8px 0"><input id="chatModelInput" placeholder="Modelo chat"/><input id="embedModelInput" placeholder="Modelo embeddings"/></div><textarea id="question" style="width:100%;height:120px" placeholder="Escribe tu pregunta..."></textarea><div id="answer" style="margin-top:8px;border:1px solid #7c4f18;padding:8px">Esperando consulta...</div><script>console.warn('UI fallback cargada');</script></body></html>"""


def _load_index_html() -> str:
    if not INDEX_HTML_PATH.exists():
        return DEFAULT_INDEX_HTML
    try:
        html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        return DEFAULT_INDEX_HTML

    required_tokens = ('id="question"', 'id="chatModelInput"', 'id="embedModelInput"', '/api/models/configure')
    if not all(token in html for token in required_tokens):
        logger.warning("Plantilla UI incompleta/corrupta; usando fallback")
        return DEFAULT_INDEX_HTML
    return html


INDEX_HTML = _load_index_html()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, status: int = HTTPStatus.OK) -> None:
        body = file_path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(file_path))
        self.send_response(status)
        self.send_header("Content-Type", (ctype or "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("Payload demasiado grande")
        raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        return json.loads(raw)

    def _get_backend(self):
        try:
            from ops_backup_benchmark import (
                export_configuration,
                get_benchmark_results,
                import_configuration,
                list_backups,
                restore_backup,
                run_backup,
                run_model_benchmark,
                start_backup_scheduler,
            )
            from profiles_config import (
                apply_model_profile,
                delete_model_profile,
                list_model_profiles,
                upsert_model_profile,
            )
            from rag_core import (
                add_web_source,
                get_data_inventory,
                get_runtime_config,
                load_or_create_vectorstore,
                rag_chat,
                set_active_models,
            )
            from state_metrics_history import (
                clear_cache,
                clear_history,
                get_cache_status,
                get_history,
                get_model_metrics,
                reload_config,
            )
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar backend RAG: {exc}") from exc

        return {
            "add_web_source": add_web_source,
            "get_data_inventory": get_data_inventory,
            "get_cache_status": get_cache_status,
            "get_history": get_history,
            "get_benchmark_results": get_benchmark_results,
            "run_model_benchmark": run_model_benchmark,
            "run_backup": run_backup,
            "list_backups": list_backups,
            "restore_backup": restore_backup,
            "start_backup_scheduler": start_backup_scheduler,
            "reload_config": reload_config,
            "get_runtime_config": get_runtime_config,
            "get_model_metrics": get_model_metrics,
            "list_model_profiles": list_model_profiles,
            "upsert_model_profile": upsert_model_profile,
            "apply_model_profile": apply_model_profile,
            "delete_model_profile": delete_model_profile,
            "export_configuration": export_configuration,
            "import_configuration": import_configuration,
            "load_or_create_vectorstore": load_or_create_vectorstore,
            "rag_chat": rag_chat,
            "set_active_models": set_active_models,
            "clear_cache": clear_cache,
            "clear_history": clear_history,
        }

    def _parse_multipart_file(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("Payload demasiado grande")
        body = self.rfile.read(content_length) if content_length else b""
        return parse_multipart_file(content_type, body)

    def do_GET(self) -> None:  # noqa: N802
        logger.info("HTTP GET request", extra={"path": self.path, "client": self.client_address[0]})
        if self.path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return

        if self.path.startswith("/assets/"):
            rel = self.path.replace("/assets/", "", 1)
            asset = (Path("assets") / rel).resolve()
            assets_root = Path("assets").resolve()
            if not str(asset).startswith(str(assets_root)) or not asset.exists() or not asset.is_file():
                self._send_json({"error": "Asset no encontrado"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_file(asset)
            return

        if self.path == "/api/config/export":
            try:
                backend = self._get_backend()
                payload = backend["export_configuration"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path in {"/api/models/profiles", "/api/profiles"}:
            try:
                backend = self._get_backend()
                profiles = backend["list_model_profiles"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"profiles": profiles})
            return

        if self.path == "/api/status":
            backend_error = None
            inventory = {"pdf_files": [], "txt_files": [], "pdf_count": 0, "txt_count": 0}
            config = {}
            model_metrics = {}
            cache_status = {}
            profiles = {}
            try:
                backend = self._get_backend()
                inventory = backend["get_data_inventory"]()
                config = backend["get_runtime_config"]()
                model_metrics = backend["get_model_metrics"]()
                cache_status = backend["get_cache_status"]()
                profiles = backend["list_model_profiles"]()
            except Exception as exc:
                backend_error = str(exc)

            with STATE.lock:
                loaded = STATE.vectorstore is not None
                metrics = STATE.metrics
                last_error = STATE.last_error
                environment_status = STATE.environment_status

            self._send_json(
                {
                    "loaded": loaded,
                    "metrics": metrics,
                    "last_error": last_error,
                    "backend_error": backend_error,
                    "inventory": inventory,
                    "config": config,
                    "model_metrics": model_metrics,
                    "cache": cache_status,
                    "model_profiles": profiles,
                    "environment": environment_status,
                }
            )
            return

        if self.path == "/api/environment/status":
            try:
                status = get_environment_status()
                with STATE.lock:
                    STATE.environment_status = status
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(status)
            return

        if self.path == "/api/cache/status":
            try:
                backend = self._get_backend()
                status = backend["get_cache_status"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(status)
            return

        if self.path.startswith("/api/history"):
            try:
                backend = self._get_backend()
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                model = params.get("model", [""])[0]
                limit_raw = params.get("limit", ["100"])[0]
                limit = int(limit_raw)
                payload = backend["get_history"](query=query, model=model, limit=limit)
            except ValueError:
                self._send_json({"error": "Parámetro limit inválido"}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path.startswith("/api/backup/list"):
            try:
                backend = self._get_backend()
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["50"])[0])
                payload = backend["list_backups"](limit=limit)
            except ValueError:
                self._send_json({"error": "Parámetro limit inválido"}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path.startswith("/api/models/benchmark/results"):
            try:
                backend = self._get_backend()
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["20"])[0])
                payload = backend["get_benchmark_results"](limit=limit)
            except ValueError:
                self._send_json({"error": "Parámetro limit inválido"}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        self._send_json({"error": "Ruta no encontrada"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        logger.info("HTTP POST request", extra={"path": self.path, "client": self.client_address[0]})
        if self.path == "/api/build":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                force_rebuild = bool(data.get("force_rebuild", True))
                vectorstore, metrics = backend["load_or_create_vectorstore"](
                    force_rebuild=force_rebuild
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                with STATE.lock:
                    STATE.last_error = str(exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            with STATE.lock:
                STATE.vectorstore = vectorstore
                STATE.metrics = metrics
                STATE.last_error = None

            mode = "reindexado" if force_rebuild else "cargado/reutilizado"
            self._send_json(
                {
                    "message": f"Índice {mode} con éxito. Métricas: {metrics or 'sin rebuild'}",
                    "metrics": metrics,
                }
            )
            return

        if self.path == "/api/upload-file":
            try:
                filename, content = self._parse_multipart_file()
                safe_name = Path(filename).name
                if not safe_name.lower().endswith((".pdf", ".txt")):
                    raise ValueError("Solo se permiten archivos .pdf o .txt")
                data_dir = Path("./data")
                data_dir.mkdir(parents=True, exist_ok=True)
                target = data_dir / safe_name
                target.write_bytes(content)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": f"No se pudo subir archivo: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json({"message": f"Archivo guardado en {target}. Reindexa para usarlo."})
            return

        if self.path == "/api/add-url":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                url = str(data.get("url", "")).strip()
                if not url:
                    raise ValueError("URL vacía")
                added = backend["add_web_source"](url)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json({"message": f"URL agregada: {added}. Reindexa para usarla."})
            return

        if self.path == "/api/models/configure":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                auto_detect = bool(data.get("auto_detect", False))
                chat_model = str(data.get("chat_model", "")).strip() or None
                embed_model = str(data.get("embed_model", "")).strip() or None
                result = backend["set_active_models"](
                    chat_model=chat_model,
                    embed_model=embed_model,
                    auto_detect=auto_detect,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json(result)
            return

        if self.path == "/api/models/benchmark":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                prompts = data.get("prompts") if isinstance(data.get("prompts"), list) else None
                models = data.get("models") if isinstance(data.get("models"), list) else None
                limit_models = int(data.get("limit_models", 5))
                payload = backend["run_model_benchmark"](
                    prompts=prompts,
                    models=models,
                    limit_models=limit_models,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path == "/api/backup/run":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                reason = str(data.get("reason", "manual")).strip() or "manual"
                payload = backend["run_backup"](reason=reason)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path == "/api/backup/restore":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                backup_id = str(data.get("backup_id", "")).strip()
                overwrite_profiles = bool(data.get("overwrite_profiles", False))
                if not backup_id:
                    raise ValueError("backup_id es requerido")
                payload = backend["restore_backup"](backup_id=backup_id, overwrite_profiles=overwrite_profiles)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path == "/api/config/import":
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                overwrite = bool(data.get("overwrite_profiles", False))
                payload = data.get("configuration") if isinstance(data.get("configuration"), dict) else data
                result = backend["import_configuration"](payload, overwrite_profiles=overwrite)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"import_result": result})
            return

        if self.path == "/api/config/reload":
            try:
                backend = self._get_backend()
                result = backend["reload_config"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return


        if self.path in {"/api/models/profile/save", "/api/profiles"}:
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                saved = backend["upsert_model_profile"](
                    name=str(data.get("name", "")).strip(),
                    chat_model=str(data.get("chat_model", "")).strip(),
                    embed_model=str(data.get("embed_model", "")).strip(),
                    description=str(data.get("description", "")).strip(),
                    generation=data.get("generation") if isinstance(data.get("generation"), dict) else None,
                    rag=data.get("rag") if isinstance(data.get("rag"), dict) else None,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"profile": saved})
            return

        if self.path in {"/api/models/profile/apply", "/api/profiles/apply"}:
            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                applied = backend["apply_model_profile"](str(data.get("name", "")).strip())
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"applied": applied})
            return

        if self.path == "/api/ask":
            with STATE.lock:
                vectorstore = STATE.vectorstore

            if vectorstore is None:
                self._send_json(
                    {"error": "Primero carga o construye el índice desde la UI."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                backend = self._get_backend()
                data = self._parse_json_body()
                question = str(data.get("question", "")).strip()
                if not question:
                    raise ValueError("La pregunta no puede estar vacía")
                answer = backend["rag_chat"](question, vectorstore)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                with STATE.lock:
                    STATE.last_error = str(exc)
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json({"answer": answer})
            return

        if self.path == "/api/cache/clear":
            try:
                backend = self._get_backend()
                payload = backend["clear_cache"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        self._send_json({"error": "Ruta no encontrada"}, status=HTTPStatus.NOT_FOUND)



    def do_DELETE(self) -> None:  # noqa: N802
        logger.info("HTTP DELETE request", extra={"path": self.path, "client": self.client_address[0]})
        if self.path == "/api/history":
            try:
                backend = self._get_backend()
                payload = backend["clear_history"]()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return

        if self.path.startswith("/api/profiles/"):
            try:
                backend = self._get_backend()
                profile_name = self.path.replace("/api/profiles/", "", 1).strip()
                if not profile_name:
                    raise ValueError("Nombre de perfil vacío")
                backend["delete_model_profile"](profile_name)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"deleted": profile_name})
            return

        self._send_json({"error": "Ruta no encontrada"}, status=HTTPStatus.NOT_FOUND)

def main() -> None:
    try:
        startup_environment = get_environment_status()
        with STATE.lock:
            STATE.environment_status = startup_environment
        if startup_environment.get("ok"):
            logger.info("Environment validation passed at startup")
        else:
            logger.warning(
                "Environment validation reported issues at startup",
                extra={
                    "errors": startup_environment.get("errors", []),
                    "warnings": startup_environment.get("warnings", []),
                },
            )
    except Exception as exc:
        logger.error("Environment validation failed at startup", extra={"error": str(exc)})

    try:
        from ops_backup_benchmark import start_backup_scheduler

        scheduler_state = start_backup_scheduler()
        logger.info("Backup scheduler state", extra=scheduler_state)
    except Exception as exc:
        logger.warning("No se pudo iniciar backup scheduler", extra={"error": str(exc)})

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    logger.info("UI server started", extra={"host": HOST, "port": PORT})
    server.serve_forever()


if __name__ == "__main__":
    main()
