"""Interfaz web local para el RAG (sin dependencias web extra)."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 7860
MAX_REQUEST_BYTES = 20_000


class AppState:
    def __init__(self) -> None:
        self.vectorstore = None
        self.metrics = None
        self.last_error = None
        self.lock = threading.Lock()


STATE = AppState()

INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Studio Pro</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(255, 255, 255, 0.08);
      --panel-border: rgba(255, 255, 255, 0.16);
      --accent: #67e8f9;
      --accent-2: #a78bfa;
      --text: #ecfeff;
      --muted: #cbd5e1;
      --ok: #22c55e;
      --warn: #f59e0b;
      --error: #ef4444;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1000px 500px at 10% -10%, #1d4ed8 0%, transparent 60%),
        radial-gradient(900px 500px at 90% 10%, #7c3aed 0%, transparent 55%),
        var(--bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .app {
      width: min(1060px, 100%);
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 20px;
      backdrop-filter: blur(10px);
      box-shadow: 0 16px 60px rgba(0, 0, 0, 0.35);
      overflow: hidden;
    }

    .header {
      padding: 24px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    h1 { margin: 0; font-size: clamp(1.4rem, 2vw, 2rem); }

    .badge {
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .content {
      padding: 24px;
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 16px;
    }

    @media (max-width: 920px) { .content { grid-template-columns: 1fr; } }

    .card {
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: rgba(2, 6, 23, 0.4);
      padding: 14px;
    }

    .row { display: flex; gap: 10px; flex-wrap: wrap; }

    textarea {
      width: 100%; min-height: 120px; border-radius: 12px;
      border: 1px solid var(--panel-border); background: rgba(2, 6, 23, 0.6);
      color: var(--text); padding: 14px; resize: vertical; outline: none; font-size: 1rem;
    }

    button {
      appearance: none; border: 0; border-radius: 12px; padding: 12px 16px;
      font-weight: 600; cursor: pointer; transition: transform .1s ease, opacity .2s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: .6; cursor: not-allowed; transform: none; }

    .btn-primary { color: #001018; background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
    .btn-secondary { background: rgba(255,255,255,0.08); color: var(--text); border: 1px solid var(--panel-border); }

    .result {
      border-radius: 12px; border: 1px solid var(--panel-border);
      background: rgba(2, 6, 23, 0.45); padding: 14px; min-height: 160px;
      white-space: pre-wrap; color: #e2e8f0; line-height: 1.55;
    }

    .status { font-weight: 600; min-height: 22px; }
    .ok { color: var(--ok); } .warn { color: var(--warn); } .error { color: var(--error); }

    .list { margin: 0; padding-left: 20px; color: #dbeafe; line-height: 1.45; }
    .tiny { font-size: .85rem; color: var(--muted); }
  </style>
</head>
<body>
  <main class="app">
    <section class="header">
      <h1>🧠 RAG Studio Pro</h1>
      <span class="badge">Auditoría + Operación · Web + PDF + TXT · Ollama + Chroma</span>
    </section>

    <section class="content">
      <section class="card">
        <div class="row">
          <button id="buildBtn" class="btn-secondary">🔧 Reindexar base</button>
          <button id="loadBtn" class="btn-secondary">📦 Cargar índice existente</button>
          <button id="exampleBtn" class="btn-secondary">💡 Pregunta ejemplo</button>
          <button id="askBtn" class="btn-primary">🚀 Preguntar</button>
        </div>

        <p id="status" class="status warn">Primero carga o construye el índice.</p>
        <textarea id="question" placeholder="Escribe tu pregunta (Ctrl + Enter para enviar)..."></textarea>
        <div class="result" id="answer">Aquí verás la respuesta del asistente...</div>
      </section>

      <aside class="card">
        <h3 style="margin-top:0">Panel de auditoría</h3>
        <p class="tiny">Estado del backend, inventario de fuentes y configuración activa.</p>
        <div id="audit" class="result" style="min-height:220px">Cargando estado...</div>
        <h4>Historial local (últimas preguntas)</h4>
        <ul id="history" class="list"></ul>
      </aside>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const questionEl = document.getElementById('question');
    const buildBtn = document.getElementById('buildBtn');
    const loadBtn = document.getElementById('loadBtn');
    const askBtn = document.getElementById('askBtn');
    const exampleBtn = document.getElementById('exampleBtn');
    const auditEl = document.getElementById('audit');
    const historyEl = document.getElementById('history');

    const history = [];

    function setStatus(text, type='warn') {
      statusEl.textContent = text;
      statusEl.className = `status ${type}`;
    }

    function toggleBusy(isBusy) {
      buildBtn.disabled = isBusy;
      loadBtn.disabled = isBusy;
      askBtn.disabled = isBusy;
      exampleBtn.disabled = isBusy;
    }

    function pushHistory(question) {
      history.unshift(question);
      if (history.length > 5) history.pop();
      historyEl.innerHTML = history.map(q => `<li>${q}</li>`).join('');
    }

    function renderAudit(data) {
      auditEl.textContent = JSON.stringify(data, null, 2);
    }

    async function refreshStatus() {
      try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        renderAudit(data);
      } catch (err) {
        auditEl.textContent = 'No se pudo cargar estado: ' + String(err);
      }
    }

    async function callBuild(force_rebuild) {
      toggleBusy(true);
      setStatus(force_rebuild ? 'Reindexando...' : 'Cargando índice existente...', 'warn');
      answerEl.textContent = 'Procesando documentos/vectorstore...';
      try {
        const resp = await fetch('/api/build', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force_rebuild })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Error en build');
        setStatus('Índice listo. Ya puedes preguntar.', 'ok');
        answerEl.textContent = data.message;
        await refreshStatus();
      } catch (err) {
        setStatus('Error al preparar índice.', 'error');
        answerEl.textContent = String(err);
      } finally {
        toggleBusy(false);
      }
    }

    exampleBtn.addEventListener('click', () => {
      questionEl.value = '¿Cuáles son los riesgos más importantes de la inteligencia artificial hoy?';
      setStatus('Pregunta ejemplo cargada.', 'ok');
    });

    buildBtn.addEventListener('click', () => callBuild(true));
    loadBtn.addEventListener('click', () => callBuild(false));

    askBtn.addEventListener('click', async () => {
      const question = questionEl.value.trim();
      if (!question) {
        setStatus('Escribe una pregunta antes de enviar.', 'warn');
        return;
      }

      toggleBusy(true);
      setStatus('Consultando al asistente...', 'warn');
      answerEl.textContent = 'Pensando...';
      pushHistory(question);

      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Error consultando al asistente');
        setStatus('Respuesta generada.', 'ok');
        answerEl.textContent = data.answer;
      } catch (err) {
        setStatus('Falló la consulta al asistente.', 'error');
        answerEl.textContent = String(err);
      } finally {
        toggleBusy(false);
      }
    });

    questionEl.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter') askBtn.click();
    });

    refreshStatus();
  </script>
</body>
</html>
"""


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

    def _parse_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("Payload demasiado grande")
        raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        return json.loads(raw)

    def _get_backend(self):
        try:
            from rag_mejorado import (
                RAGSetupError,
                get_data_inventory,
                get_runtime_config,
                load_or_create_vectorstore,
                rag_chat,
            )
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar backend RAG: {exc}") from exc

        return {
            "RAGSetupError": RAGSetupError,
            "get_data_inventory": get_data_inventory,
            "get_runtime_config": get_runtime_config,
            "load_or_create_vectorstore": load_or_create_vectorstore,
            "rag_chat": rag_chat,
        }

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return

        if self.path == "/api/status":
            backend_error = None
            inventory = {"pdf_files": [], "txt_files": [], "pdf_count": 0, "txt_count": 0}
            config = {}
            try:
                backend = self._get_backend()
                inventory = backend["get_data_inventory"]()
                config = backend["get_runtime_config"]()
            except Exception as exc:
                backend_error = str(exc)

            with STATE.lock:
                loaded = STATE.vectorstore is not None
                metrics = STATE.metrics
                last_error = STATE.last_error

            self._send_json(
                {
                    "loaded": loaded,
                    "metrics": metrics,
                    "last_error": last_error,
                    "backend_error": backend_error,
                    "inventory": inventory,
                    "config": config,
                }
            )
            return

        self._send_json({"error": "Ruta no encontrada"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
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

        self._send_json({"error": "Ruta no encontrada"}, status=HTTPStatus.NOT_FOUND)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"RAG Studio Pro disponible en http://{HOST}:{PORT}")
    print("Presiona Ctrl+C para detener.")
    server.serve_forever()


if __name__ == "__main__":
    main()
