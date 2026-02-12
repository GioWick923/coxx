"""Interfaz web local para el RAG (sin dependencias web extra)."""

from __future__ import annotations

import cgi
import json
import mimetypes
import threading
from pathlib import Path
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
  <title>RAG Studio // Cyber Deck</title>
  <style>
    :root {
      --bg-0: #09070a;
      --bg-1: #130d08;
      --bg-2: #1d1208;
      --ink: #ffd56a;
      --ink-soft: #ffbf3b;
      --line: #5b3d18;
      --line-2: #2f1f0d;
      --panel: rgba(20, 11, 5, 0.94);
      --ok: #70ffa8;
      --warn: #ffc857;
      --error: #ff6b6b;
      --cyan: #59f0ff;
      --shadow: 0 0 22px rgba(255, 180, 60, .2);
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 8%, rgba(255,190,70,.15), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(89,240,255,.08), transparent 25%),
        linear-gradient(160deg, var(--bg-0), var(--bg-1) 42%, var(--bg-2)),
        url("/assets/cyber_bg.svg");
      background-size: auto, auto, auto, cover;
      background-attachment: fixed;
      font-family: "Orbitron", "Share Tech Mono", "Rajdhani", "Consolas", monospace;
      letter-spacing: .25px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }

    .screen {
      width: min(1280px, 100%);
      min-height: 92vh;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(40,22,8,.92), rgba(16,10,6,.96));
      box-shadow: var(--shadow), inset 0 0 0 1px rgba(255,200,100,.1);
      border-radius: 16px;
      overflow: hidden;
      position: relative;
    }

    .scanline {
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(to bottom, rgba(255,255,255,.02) 1px, transparent 1px);
      background-size: 100% 4px;
      opacity: .18;
      mix-blend-mode: soft-light;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(20,11,5,.86);
    }

    .brand {
      font-size: .95rem;
      color: var(--ink-soft);
      text-transform: uppercase;
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .pill {
      border: 1px solid var(--line);
      padding: 4px 8px;
      border-radius: 999px;
      font-size: .75rem;
      color: #ffd98a;
      background: rgba(255,170,50,.06);
    }

    .layout {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 14px;
      padding: 14px;
    }

    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
    }

    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 10px;
      padding: 12px;
      box-shadow: inset 0 0 0 1px rgba(255, 214, 106, .06);
    }

    .panel-title {
      margin: 0 0 10px;
      font-size: .92rem;
      color: var(--ink-soft);
      text-transform: uppercase;
      letter-spacing: .8px;
    }

    .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }

    button {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #2a190a, #1a1108);
      color: var(--ink);
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 700;
      font-size: .8rem;
      text-transform: uppercase;
      cursor: pointer;
      transition: transform .08s ease, border-color .2s ease, box-shadow .2s ease;
    }

    button:hover {
      transform: translateY(-1px);
      border-color: #8a5a1f;
      box-shadow: 0 0 0 1px rgba(255,196,88,.15);
    }

    button:disabled { opacity: .5; cursor: not-allowed; transform: none; }

    .btn-primary {
      background: linear-gradient(180deg, #ffbf3b, #b16e14);
      color: #120b05;
      border-color: #f2a425;
      text-shadow: none;
    }

    input[type="text"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #120b07;
      color: #ffe3a1;
      padding: 10px;
      outline: none;
      font-family: "Share Tech Mono", "Consolas", monospace;
      font-size: .9rem;
    }

    input[type="text"]::placeholder, textarea::placeholder { color: #936937; }
    textarea { min-height: 120px; resize: vertical; }
    input[type="file"] { color: #e8bc62; font-size: .86rem; }

    .status { min-height: 20px; font-size: .85rem; margin: 6px 0 10px; }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .error { color: var(--error); }

    .result {
      border: 1px solid var(--line);
      background: rgba(10,7,5,.9);
      border-radius: 8px;
      padding: 10px;
      min-height: 160px;
      white-space: pre-wrap;
      line-height: 1.5;
      color: #ffd78d;
      font-size: .86rem;
    }

    .util-row { display:flex; gap:8px; justify-content:flex-end; margin: 8px 0 6px; }

    

    .icon-anim {
      width: 18px;
      height: 18px;
      vertical-align: middle;
      margin-right: 6px;
      filter: drop-shadow(0 0 6px rgba(255, 200, 90, .3));
    }

    .status-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 4px;
    }

    .status-anim {
      width: 24px;
      height: 24px;
      opacity: .9;
    }
.tiny { margin: 0 0 10px; color: #c39145; font-size: .78rem; }
    .hint { margin: 6px 0 8px; color: #cf9f52; font-size: .78rem; }
    .list { margin: 0; padding-left: 18px; font-size: .82rem; color: #f1c36e; }

    .accent-cyan {
      color: var(--cyan);
      text-shadow: 0 0 10px rgba(89,240,255,.35);
      font-size: .8rem;
    }


    .table-wrap {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      background: rgba(12, 8, 5, .9);
    }

    table.models-table {
      width: 100%;
      border-collapse: collapse;
      font-size: .78rem;
      min-width: 580px;
    }

    .models-table th, .models-table td {
      border-bottom: 1px solid var(--line-2);
      padding: 8px 10px;
      text-align: left;
      color: #ffd78d;
    }

    .models-table th {
      position: sticky;
      top: 0;
      background: #1d130a;
      color: var(--ink-soft);
      text-transform: uppercase;
      font-size: .72rem;
      letter-spacing: .6px;
    }

    .chip {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: .7rem;
      color: #ffcf77;
      background: rgba(255, 190, 70, .08);
    }
  </style>
</head>
<body>
  <main class="screen">
    <div class="scanline"></div>

    <header class="topbar">
      <div class="brand" title="Consola principal de operación del RAG local">
        <span>⚡ RAG Studio Cyber Deck</span>
        <span class="accent-cyan">v.local</span>
      </div>
      <span class="pill" title="UI local con backend Python y persistencia en Chroma">PDF / TXT / WEB</span>
    </header>

    <section class="layout">
      <section class="panel">
        <h3 class="panel-title" title="Acciones para cargar índice y consultar"><img class="icon-anim" src="/assets/anim_scan.svg" alt="scan"/>Control de misión</h3>

        <div class="row">
          <button id="buildBtn" title="Reconstruye el índice completo con todas las fuentes detectadas">🔧 Reindexar</button>
          <button id="loadBtn" title="Carga el índice existente sin recrearlo">📦 Cargar índice</button>
          <button id="exampleBtn" title="Inserta una pregunta de ejemplo">💡 Ejemplo</button>
          <button id="askBtn" class="btn-primary" title="Envía la pregunta al backend RAG">🚀 Preguntar</button>
        </div>

        <div class="panel" title="Selecciona modelos Ollama disponibles en tu PC">
          <h3 class="panel-title" style="margin-bottom:6px"><img class="icon-anim" src="/assets/anim_loader.svg" alt="models"/>Modelos Ollama</h3>
          <p class="hint">Puedes elegir manualmente o usar auto-detección inteligente de modelos instalados localmente.</p>
          <div class="row">
            <input id="chatModelInput" type="text" placeholder="Modelo chat (ej: llama3.1:8b)" title="Modelo LLM para generar respuestas" />
            <input id="embedModelInput" type="text" placeholder="Modelo embeddings (ej: nomic-embed-text)" title="Modelo para vectorizar documentos" />
          </div>
          <div class="row">
            <button id="refreshModelsBtn" title="Consultar modelos detectados por ollama list">🔄 Detectar modelos</button>
            <button id="saveModelsBtn" title="Guardar modelos indicados y aplicarlos al backend">💾 Aplicar modelos</button>
            <button id="autoModelsBtn" title="Elegir automáticamente modelos recomendados según los instalados">⚙️ Auto-configurar</button>
          </div>
        </div>

        <div class="panel" title="Anexa documentos o links sin salir de la interfaz">
          <h3 class="panel-title" style="margin-bottom:6px"><img class="icon-anim" src="/assets/anim_loader.svg" alt="load"/>Ingesta de fuentes</h3>
          <p class="hint">Sube archivos <b>.pdf/.txt</b> o agrega una URL para incorporar al corpus. Luego reindexa.</p>
          <div class="row">
            <input id="fileInput" type="file" accept=".pdf,.txt" title="Selecciona PDF/TXT local" />
            <button id="uploadBtn" title="Guarda el archivo en la carpeta data">📎 Subir archivo</button>
          </div>
          <div class="row">
            <input id="urlInput" type="text" placeholder="https://..." title="URL de documento/artículo para ingestión" />
            <button id="addUrlBtn" title="Guarda la URL en el catálogo web_sources.txt">🔗 Agregar link</button>
          </div>
        </div>

        <div class="status-row" title="Estado de la última operación">
          <img id="statusAnim" class="status-anim" src="/assets/anim_scan.svg" alt="status"/>
          <p id="status" class="status warn">Cargar o reindexar para empezar.</p>
        </div>
        <textarea id="question" placeholder="Escribe tu pregunta... (Ctrl + Enter para enviar)" title="Pregunta para el asistente"></textarea>
        <div class="util-row">
          <button id="copyBtn" title="Copiar respuesta actual al portapapeles"><img class="icon-anim" src="/assets/anim_scan.svg" alt="copy"/>Copiar</button>
          <button id="saveBtn" title="Guardar respuesta actual como archivo TXT"><img class="icon-anim" src="/assets/anim_save.svg" alt="save"/>Guardar TXT</button>
        </div>
        <div id="answer" class="result" title="Respuesta del modelo y mensajes del flujo">Esperando consulta...</div>
        <p id="answerMeta" class="tiny" title="Información breve de la respuesta">Sin respuesta generada todavía.</p>
      </section>

      <aside class="panel">
        <h3 class="panel-title" title="Métricas, inventario y estado interno"><img class="icon-anim" src="/assets/anim_scan.svg" alt="telemetry"/>Auditoría y telemetría</h3>
        <p class="tiny">Estado backend · inventario de fuentes · config activa.</p>
        <div id="audit" class="result" style="min-height:220px" title="Estado JSON del sistema">Cargando estado...</div>
        <h3 class="panel-title" style="margin-top:12px" title="Tabla comparativa de modelos detectados">Comparativa de modelos</h3>
        <div class="table-wrap" title="Nombre, latencia promedio, RAM estimada y tipo de tarea">
          <table class="models-table">
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Latencia promedio</th>
                <th>RAM estimada</th>
                <th>Tipo de tarea</th>
              </tr>
            </thead>
            <tbody id="modelsTableBody">
              <tr><td colspan="4">Cargando métricas de modelos...</td></tr>
            </tbody>
          </table>
        </div>
        <h3 class="panel-title" style="margin-top:12px" title="Preguntas recientes en esta sesión">Historial local</h3>
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
    const uploadBtn = document.getElementById('uploadBtn');
    const addUrlBtn = document.getElementById('addUrlBtn');
    const copyBtn = document.getElementById('copyBtn');
    const saveBtn = document.getElementById('saveBtn');
    const fileInput = document.getElementById('fileInput');
    const urlInput = document.getElementById('urlInput');
    const auditEl = document.getElementById('audit');
    const historyEl = document.getElementById('history');
    const answerMetaEl = document.getElementById('answerMeta');
    const statusAnimEl = document.getElementById('statusAnim');
    const modelsTableBody = document.getElementById('modelsTableBody');
    const chatModelInput = document.getElementById('chatModelInput');
    const embedModelInput = document.getElementById('embedModelInput');
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');
    const saveModelsBtn = document.getElementById('saveModelsBtn');
    const autoModelsBtn = document.getElementById('autoModelsBtn');

    const history = [];

    function setStatus(text, type='warn') {
      statusEl.textContent = text;
      statusEl.className = `status ${type}`;
      const map = { ok: '/assets/anim_save.svg', warn: '/assets/anim_loader.svg', error: '/assets/anim_scan.svg' };
      statusAnimEl.src = map[type] || '/assets/anim_loader.svg';
    }

    function toggleBusy(isBusy) {
      for (const b of [buildBtn, loadBtn, askBtn, exampleBtn, uploadBtn, addUrlBtn, copyBtn, saveBtn, refreshModelsBtn, saveModelsBtn, autoModelsBtn]) {
        b.disabled = isBusy;
      }
    }

    function pushHistory(question) {
      history.unshift(question);
      if (history.length > 5) history.pop();
      historyEl.innerHTML = history.map(q => `<li>${q}</li>`).join('');
    }

    

    function updateAnswerMeta(text) {
      const clean = (text || '').trim();
      if (!clean) {
        answerMetaEl.textContent = 'Sin respuesta generada todavía.';
        return;
      }
      const chars = clean.length;
      const lines = clean.split(/\n+/).filter(Boolean).length;
      const now = new Date().toLocaleTimeString();
      answerMetaEl.textContent = `Longitud: ${chars} caracteres · ${lines} líneas · ${now}`;
    }

    async function copyAnswer() {
      const text = (answerEl.textContent || '').trim();
      if (!text || text === 'Esperando consulta...' || text === 'Pensando...') {
        setStatus('No hay una respuesta válida para copiar.', 'warn');
        return;
      }
      try {
        await navigator.clipboard.writeText(text);
        setStatus('Respuesta copiada al portapapeles.', 'ok');
      } catch (err) {
        setStatus('No se pudo copiar automáticamente.', 'error');
      }
    }

    function saveAnswerAsTxt() {
      const text = (answerEl.textContent || '').trim();
      if (!text || text === 'Esperando consulta...' || text === 'Pensando...') {
        setStatus('No hay una respuesta válida para guardar.', 'warn');
        return;
      }
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const fileName = `respuesta_rag_${ts}.txt`;
      const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatus('Respuesta guardada como archivo TXT.', 'ok');
    }


    function fillModelInputs(config = {}) {
      if (chatModelInput) chatModelInput.value = config.chat_model || chatModelInput.value || '';
      if (embedModelInput) embedModelInput.value = config.embed_model || embedModelInput.value || '';
    }

    async function configureModels(auto_detect=false) {
      toggleBusy(true);
      setStatus(auto_detect ? 'Auto-configurando modelos...' : 'Aplicando modelos...', 'warn');
      try {
        const payload = {
          auto_detect,
          chat_model: (chatModelInput.value || '').trim(),
          embed_model: (embedModelInput.value || '').trim(),
        };
        const resp = await fetch('/api/models/configure', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'No se pudo configurar modelos');
        fillModelInputs(data);
        setStatus(`Modelos activos: chat=${data.chat_model} | embed=${data.embed_model}`, 'ok');
        await refreshStatus();
      } catch (err) {
        setStatus('Error configurando modelos.', 'error');
        answerEl.textContent = String(err);
        updateAnswerMeta(String(err));
      } finally {
        toggleBusy(false);
      }
    }


    function estimateModelRamMB(modelName, metrics) {
      const total = metrics?.memory?.rss_mb;
      if (typeof total !== 'number') return null;
      const activeChat = metrics?.active_models?.chat?.name;
      const activeEmbed = metrics?.active_models?.embedding?.name;
      if (modelName === activeChat && modelName === activeEmbed) return total;
      if (modelName === activeChat) return +(total * 0.6).toFixed(1);
      if (modelName === activeEmbed) return +(total * 0.4).toFixed(1);
      return +(total * 0.2).toFixed(1);
    }

    function renderModelsTable(statusData) {
      const metrics = statusData?.model_metrics || {};
      const detected = metrics.detected_models || [];
      const perf = metrics.performance || {};
      if (!detected.length) {
        modelsTableBody.innerHTML = '<tr><td colspan="4">No hay modelos detectados en Ollama.</td></tr>';
        return;
      }

      const rows = detected.map((m) => {
        const type = m.type || 'chat';
        const perfKey = type === 'embedding' ? 'embedding' : 'chat';
        const avg = perf?.[perfKey]?.avg_latency_ms ?? 0;
        const ram = estimateModelRamMB(m.name, metrics);
        const ramText = ram == null ? 'N/D' : `${ram} MB`;
        return `
          <tr>
            <td>${m.name}</td>
            <td>${avg} ms</td>
            <td>${ramText}</td>
            <td><span class="chip">${type}</span></td>
          </tr>
        `;
      }).join('');

      modelsTableBody.innerHTML = rows;
    }

    async function refreshStatus() {
      try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        auditEl.textContent = JSON.stringify(data, null, 2);
        fillModelInputs(data.config || {});
        renderModelsTable(data);
      } catch (err) {
        auditEl.textContent = 'No se pudo cargar estado: ' + String(err);
        modelsTableBody.innerHTML = '<tr><td colspan="4">Error cargando tabla de modelos.</td></tr>';
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
        updateAnswerMeta(data.message);
        await refreshStatus();
      } catch (err) {
        setStatus('Error al preparar índice.', 'error');
        answerEl.textContent = String(err);
        updateAnswerMeta(String(err));
      } finally {
        toggleBusy(false);
      }
    }

    async function uploadFile() {
      const file = fileInput.files[0];
      if (!file) {
        setStatus('Selecciona un archivo PDF/TXT primero.', 'warn');
        return;
      }
      toggleBusy(true);
      setStatus('Subiendo archivo...', 'warn');
      const form = new FormData();
      form.append('file', file);
      try {
        const resp = await fetch('/api/upload-file', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Error subiendo archivo');
        setStatus('Archivo agregado correctamente.', 'ok');
        answerEl.textContent = data.message;
        updateAnswerMeta(data.message);
        fileInput.value = '';
        await refreshStatus();
      } catch (err) {
        setStatus('Error al subir archivo.', 'error');
        answerEl.textContent = String(err);
        updateAnswerMeta(String(err));
      } finally {
        toggleBusy(false);
      }
    }

    async function addUrl() {
      const url = urlInput.value.trim();
      if (!url) {
        setStatus('Ingresa una URL válida.', 'warn');
        return;
      }
      toggleBusy(true);
      setStatus('Guardando URL...', 'warn');
      try {
        const resp = await fetch('/api/add-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Error agregando URL');
        setStatus('URL agregada correctamente.', 'ok');
        answerEl.textContent = data.message;
        updateAnswerMeta(data.message);
        urlInput.value = '';
        await refreshStatus();
      } catch (err) {
        setStatus('Error agregando URL.', 'error');
        answerEl.textContent = String(err);
        updateAnswerMeta(String(err));
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
    uploadBtn.addEventListener('click', uploadFile);
    addUrlBtn.addEventListener('click', addUrl);
    copyBtn.addEventListener('click', copyAnswer);
    saveBtn.addEventListener('click', saveAnswerAsTxt);
    refreshModelsBtn.addEventListener('click', refreshStatus);
    saveModelsBtn.addEventListener('click', () => configureModels(false));
    autoModelsBtn.addEventListener('click', () => configureModels(true));

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
        updateAnswerMeta(data.answer);
      } catch (err) {
        setStatus('Falló la consulta al asistente.', 'error');
        answerEl.textContent = String(err);
        updateAnswerMeta(String(err));
      } finally {
        toggleBusy(false);
      }
    });

    questionEl.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter') askBtn.click();
    });

    refreshStatus();
    updateAnswerMeta('');
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
            from rag_mejorado import (
                add_web_source,
                get_data_inventory,
                get_cache_status,
                apply_model_profile,
                delete_model_profile,
                export_configuration,
                get_model_metrics,
                get_runtime_config,
                list_model_profiles,
                load_or_create_vectorstore,
                rag_chat,
                set_active_models,
                upsert_model_profile,
                import_configuration,
                clear_cache,
            )
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar backend RAG: {exc}") from exc

        return {
            "add_web_source": add_web_source,
            "get_data_inventory": get_data_inventory,
            "get_cache_status": get_cache_status,
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
        }

    def _parse_multipart_file(self):
        ctype, _ = cgi.parse_header(self.headers.get("content-type", ""))
        if ctype != "multipart/form-data":
            raise ValueError("Se esperaba multipart/form-data")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )
        if "file" not in form:
            raise ValueError("No se encontró el campo file")
        fileitem = form["file"]
        if not fileitem.filename:
            raise ValueError("Archivo vacío")
        data = fileitem.file.read()
        return fileitem.filename, data

    def do_GET(self) -> None:  # noqa: N802
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
                }
            )
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
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"RAG Studio Pro disponible en http://{HOST}:{PORT}")
    print("Presiona Ctrl+C para detener.")
    server.serve_forever()


if __name__ == "__main__":
    main()
