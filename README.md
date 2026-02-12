# RAG Studio Pro (Local)

Proyecto RAG local con:
- `rag_mejorado.py` (pipeline RAG incremental + metadata enriquecida)
- `web_ui.py` (UI local con auditoría y carga de fuentes)
- `run_rag_ui_windows.bat` (arranque de un clic en Windows)


## Estructura modular (refactor de mantenimiento)
Se separó la lógica en módulos para reducir acoplamiento sin cambiar endpoints ni comportamiento:
- `api_server.py`: rutas HTTP y servidor API/UI
- `web_ui.py`: entry point ligero que reexporta servidor
- `rag_core.py`: fachada de operaciones núcleo RAG
- `profiles_config.py`: fachada de perfiles
- `ops_backup_benchmark.py`: fachada de backup/benchmark/config ops
- `state_metrics_history.py`: fachada de métricas/cache/historial/estado
- `ui_assets/index.html`: plantilla HTML separada del código servidor

## Requisitos
- Python 3.10+
- Ollama instalado y activo
- Modelos Ollama descargados:
  - Embeddings: `nomic-embed-text`
  - Chat: `llama3.1:8b`

## Instalación
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```


## Preflight (recomendado)
```bash
python preflight_check.py
```
Este comando valida:
- versión de Python
- dependencias instaladas
- archivos base del proyecto
- disponibilidad de Ollama y modelos requeridos
  - detección robusta por CLI, API local (`OLLAMA_HOST`) y fallback por manifiestos en carpeta de modelos (`OLLAMA_MODELS` o `~/.ollama/models`)

## Estado de entorno (Environment Validator)
El sistema ejecuta validación automática al iniciar `web_ui.py` y expone el estado en:

- `GET /api/environment/status`
- `GET /api/status` (campo `environment`)

Validaciones incluidas:
- versión mínima de Python
- dependencias Python requeridas
- disponibilidad de Ollama + modelos configurados
- acceso de escritura a disco (`data`, `chroma_db`, carpeta de logs)

### Ejemplo JSON de `/api/environment/status`
```json
{
  "ok": false,
  "timestamp": "2026-02-12T17:21:36.112000+00:00",
  "checks": {
    "python": {"ok": true, "errors": [], "warnings": [], "suggestions": [], "details": {"current": "3.11.9", "minimum": "3.10"}},
    "dependencies": {"ok": true, "errors": [], "warnings": [], "suggestions": [], "details": {"checked_modules": ["bs4", "langchain_community"]}},
    "ollama": {
      "ok": false,
      "errors": [],
      "warnings": ["Modelos configurados no encontrados: nomic-embed-text, llama3.1:8b"],
      "suggestions": ["Actualiza perfiles/modelos activos desde la UI o instala los modelos faltantes con `ollama pull`."],
      "details": {"detected_models": ["llama3.2:3b"], "missing_required_models": ["nomic-embed-text", "llama3.1:8b"]}
    },
    "disk": {"ok": true, "errors": [], "warnings": [], "suggestions": [], "details": {"writable_paths": ["data", "chroma_db", "logs"]}}
  },
  "errors": [],
  "warnings": ["Modelos configurados no encontrados: nomic-embed-text, llama3.1:8b"],
  "suggestions": ["Actualiza perfiles/modelos activos desde la UI o instala los modelos faltantes con `ollama pull`."]
}
```

## Backups automáticos y manuales
El backend incluye respaldo versionado en `chroma_db/backups/` para proteger:
- perfiles
- configuración activa/global
- historial de consultas
- métricas (snapshot de modelos + histórico de benchmark)

Endpoints:
- `POST /api/backup/run` → ejecuta backup manual
- `GET /api/backup/list` → lista backups
- `POST /api/backup/restore` → restaura backup

Config programación automática:
- `RAG_BACKUP_INTERVAL_SECONDS` (por defecto `3600`)
- usar `0` o negativo para deshabilitar scheduler automático

Ejemplos:
```bash
curl -sS -X POST http://127.0.0.1:7860/api/backup/run -H 'Content-Type: application/json' -d '{"reason":"manual"}'
curl -sS http://127.0.0.1:7860/api/backup/list
curl -sS -X POST http://127.0.0.1:7860/api/backup/restore -H 'Content-Type: application/json' -d '{"backup_id":"backup_1730000000000","overwrite_profiles":true}'
```

## Ejecutar UI local
```bash
python web_ui.py
```
Abrir: http://127.0.0.1:7860

## Flujo recomendado
1. En UI, usa **Agregar fuentes**:
   - subir `.pdf`/`.txt`
   - o agregar URL
2. Clic en **Reindexar base**.
3. Escribir pregunta y clic en **Preguntar**.

## Windows (doble clic)
Usa `run_rag_ui_windows.bat`.
- Detecta Python
- Activa `.venv` si existe
- Instala `requirements.txt`
- Lanza `web_ui.py`

## Pruebas
```bash
python -m py_compile rag_mejorado.py web_ui.py tests/test_rag_mejorado.py
python -m unittest discover -s tests -v
```

## Notas técnicas
- Fuentes web adicionales se guardan en `data/web_sources.txt`.
- Los archivos subidos desde UI se guardan en `data/`.
- El indexado incremental usa registro local en `chroma_db/index_registry.json`.


## Endpoint de métricas de modelos (`/api/status`)
`GET /api/status` ahora incluye `model_metrics` con información en tiempo real:
- **latencia promedio por request** (por tarea/modelo activo)
- **uso de memoria del proceso** (RSS MB)
- **nombre y versión** de modelos activos y detectados

### Formato JSON (resumen)
```json
{
  "loaded": true,
  "inventory": {"pdf_count": 1, "txt_count": 2, "web_count": 3},
  "config": {"chat_model": "llama3.1:8b", "embed_model": "nomic-embed-text"},
  "model_metrics": {
    "active_models": {
      "chat": {"name": "llama3.1:8b", "version": "8b"},
      "embedding": {"name": "nomic-embed-text", "version": "latest"}
    },
    "detected_models": [
      {"name": "llama3.1:8b", "version": "8b", "type": "chat"},
      {"name": "nomic-embed-text", "version": "latest", "type": "embedding"}
    ],
    "performance": {
      "chat": {"model": "llama3.1:8b", "request_count": 5, "avg_latency_ms": 132.4},
      "embedding": {"model": "nomic-embed-text", "request_count": 2, "avg_latency_ms": 47.8},
      "deep_analysis": {"model": "llama3.1:8b", "request_count": 0, "avg_latency_ms": 0.0}
    },
    "memory": {"rss_mb": 210.6, "source": "resource"}
  }
}
```

### Ejemplo rápido
```bash
curl -sS http://127.0.0.1:7860/api/status | python -m json.tool
```


## Prompts adaptativos en RAG
El backend usa plantillas de prompt según tipo de consulta y longitud de contexto:

1. **Resumen de texto largo** (`summary_long`)
   - Se activa si la consulta contiene términos como: `resumen`, `resume`, `sintetiza`.
2. **QA contexto corto/largo**
   - `qa_short` si el contexto es corto.
   - `qa_long` si el contexto supera el umbral (actual: ~2200 caracteres).
3. **Razonamiento paso a paso** (`reasoning_steps`)
   - Se activa si la consulta contiene señales como `paso a paso`, `analiza`, `explica cómo`.

### Reglas de selección (resumen)
- Prioridad 1: intención de resumen.
- Prioridad 2: intención de razonamiento paso a paso.
- Prioridad 3: QA normal según longitud de contexto (corto/largo).

### Ejemplos de prompts
- **Resumen largo**
  - Entrada: `"Haz un resumen ejecutivo del documento"`
  - Plantilla: JSON con `summary`, `key_points`, `risks_or_limits`, `sources`.
- **QA corto**
  - Entrada: `"¿Cuál es la conclusión principal?"` + contexto corto.
  - Plantilla: JSON con `answer`, `confidence`, `sources`.
- **QA largo**
  - Entrada: `"¿Qué riesgos y beneficios se comparan?"` + contexto extenso.
  - Plantilla: JSON con `answer`, `evidence`, `confidence`, `sources`.
- **Reasoning**
  - Entrada: `"Analiza paso a paso la diferencia entre enfoques"`.
  - Plantilla: JSON con `answer`, `steps`, `confidence`, `sources`.


## Tabla comparativa de modelos en UI (HTML/CSS/JS)
La UI incluye una tabla "Comparativa de modelos" que muestra:
- Nombre del modelo
- Latencia promedio
- Uso de RAM estimado
- Tipo de tarea (chat/embedding)

### Datos simulados (ejemplo)
```json
{
  "model_metrics": {
    "active_models": {
      "chat": {"name": "llama3.1:8b", "version": "8b"},
      "embedding": {"name": "nomic-embed-text", "version": "latest"}
    },
    "detected_models": [
      {"name": "llama3.1:8b", "version": "8b", "type": "chat"},
      {"name": "nomic-embed-text", "version": "latest", "type": "embedding"},
      {"name": "qwen2.5:14b", "version": "14b", "type": "chat"}
    ],
    "performance": {
      "chat": {"model": "llama3.1:8b", "request_count": 12, "avg_latency_ms": 142.3},
      "embedding": {"model": "nomic-embed-text", "request_count": 4, "avg_latency_ms": 49.1}
    },
    "memory": {"rss_mb": 512.0, "source": "resource"}
  }
}
```

### Cómo consume JSON del backend
1. `refreshStatus()` llama `GET /api/status`.
2. De la respuesta usa `model_metrics.detected_models` para las filas.
3. Usa `model_metrics.performance.chat/embedding.avg_latency_ms` para latencia promedio.
4. Usa `model_metrics.memory.rss_mb` para estimar RAM por modelo (heurística UI).
5. Renderiza en `<tbody id="modelsTableBody">`.


## Perfiles de configuración de modelos
Se añadió soporte para perfiles (JSON) para alternar rápidamente entre configuraciones como:
- **Rápido**
- **Preciso**
- **Bajo consumo**

### 1) Estructura JSON de perfiles
Archivo: `chroma_db/model_profiles.json`

```json
{
  "Rapido": {
    "description": "Prioriza velocidad de respuesta con modelos livianos.",
    "chat_model": "llama3.1:8b",
    "embed_model": "nomic-embed-text"
  },
  "Preciso": {
    "description": "Prioriza calidad/razonamiento con modelos más robustos.",
    "chat_model": "deepseek-r1:32b",
    "embed_model": "nomic-embed-text"
  },
  "Bajo consumo": {
    "description": "Minimiza uso de recursos para equipos modestos.",
    "chat_model": "phi3:mini",
    "embed_model": "nomic-embed-text"
  }
}
```

### 2) Funciones backend
- `list_model_profiles()` → lista perfiles guardados.
- `upsert_model_profile(name, chat_model, embed_model, description)` → crea/edita perfil.
- `apply_model_profile(name)` → aplica perfil y actualiza modelos activos.

### 3) Endpoints UI/backend para perfiles
- `GET /api/models/profiles` → retorna perfiles.
- `POST /api/models/profile/save` → crea/edita perfil.
- `POST /api/models/profile/apply` → aplica perfil y activa modelos.

Ejemplo guardar perfil:
```bash
curl -sS -X POST http://127.0.0.1:7860/api/models/profile/save   -H 'Content-Type: application/json'   -d '{"name":"Demo","chat_model":"llama3.1:8b","embed_model":"nomic-embed-text","description":"perfil demo"}'
```

Ejemplo aplicar perfil:
```bash
curl -sS -X POST http://127.0.0.1:7860/api/models/profile/apply   -H 'Content-Type: application/json'   -d '{"name":"Rapido"}'
```

### Uso desde UI
- La UI puede cargar perfiles desde `/api/models/profiles` para poblar un selector.
- Al elegir uno, llamar `/api/models/profile/apply`.
- Para crear/editar, enviar formulario a `/api/models/profile/save`.
- El estado actualizado (modelos activos + perfiles) queda visible en `/api/status`.


### Endpoints estándar de perfiles (recomendados)
Además de rutas legacy (`/api/models/...`), se soportan estas rutas principales:
- `GET /api/profiles`
- `POST /api/profiles`
- `DELETE /api/profiles/{name}`
- `POST /api/profiles/apply`

#### Ejemplo JSON completo de perfil
```json
{
  "name": "PrecisoCustom",
  "description": "Mejor calidad para análisis complejos",
  "chat_model": "deepseek-r1:32b",
  "embed_model": "nomic-embed-text",
  "generation": {
    "temperature": 0.05,
    "top_p": 0.95
  },
  "rag": {
    "top_k": 5,
    "max_chars_per_chunk": 1200,
    "max_total_context_chars": 4600
  }
}
```

#### Documentación básica de endpoints
- `GET /api/profiles`
  - Respuesta: `{ "profiles": { ... } }`
- `POST /api/profiles`
  - Body: JSON del perfil (como el ejemplo)
  - Acción: crea o edita perfil por `name`
- `DELETE /api/profiles/{name}`
  - Acción: elimina el perfil indicado
- `POST /api/profiles/apply`
  - Body: `{ "name": "Rapido" }`
  - Acción: valida modelos en Ollama y aplica modelos + configuración runtime (`generation`, `rag`)


## Exportar / Importar configuración global
Permite migrar y compartir setups completos del sistema.

### Endpoints
- `GET /api/config/export`
- `POST /api/config/import`

### ¿Qué exporta?
Un solo JSON con:
- perfiles existentes
- modelos activos actuales
- configuración runtime relevante

### Ejemplo de archivo exportado
```json
{
  "schema_version": 1,
  "exported_at_ms": 1760000000000,
  "profiles": {
    "Rapido": {
      "description": "Prioriza velocidad de respuesta con modelos livianos.",
      "chat_model": "llama3.1:8b",
      "embed_model": "nomic-embed-text",
      "generation": {"temperature": 0.2, "top_p": 0.85},
      "rag": {"top_k": 3, "max_chars_per_chunk": 900, "max_total_context_chars": 3200}
    }
  },
  "active": {
    "chat_model": "llama3.1:8b",
    "embed_model": "nomic-embed-text",
    "generation": {"temperature": 0.1, "top_p": 0.9},
    "rag": {"top_k": 4, "max_chars_per_chunk": 1200, "max_total_context_chars": 4200}
  },
  "runtime": {
    "chat_model": "llama3.1:8b",
    "embed_model": "nomic-embed-text"
  }
}
```

### Validaciones al importar
1. Integridad del archivo (`schema_version`, `profiles`, `active`).
2. Tipos/estructura de perfiles.
3. Existencia de modelos en Ollama.
4. Conflictos de nombres de perfiles:
   - por defecto **no sobrescribe** y devuelve error,
   - usar `overwrite_profiles=true` para confirmar sobrescritura.

### Ejemplos
Exportar:
```bash
curl -sS http://127.0.0.1:7860/api/config/export > config_export.json
```

Importar sin sobrescribir:
```bash
curl -sS -X POST http://127.0.0.1:7860/api/config/import   -H 'Content-Type: application/json'   -d @config_export.json
```

Importar con sobrescritura confirmada:
```bash
curl -sS -X POST http://127.0.0.1:7860/api/config/import   -H 'Content-Type: application/json'   -d '{"overwrite_profiles": true, "configuration": { ... }}'
```

## Migración parser multipart (sin `cgi`)

Para compatibilidad con Python 3.13+ se eliminó el parser basado en `cgi.FieldStorage` y se reemplazó por un parser multipart moderno usando librería estándar `email`.

### Cambio técnico
- Antes: `cgi.parse_header` + `cgi.FieldStorage`.
- Ahora: `email.parser.BytesParser` sobre un mensaje MIME sintético con el `Content-Type` original.

### Ventajas
- Evita `DeprecationWarning` y futura rotura por eliminación de `cgi`.
- Mantiene contrato del endpoint `POST /api/upload-file` (campo `file`, mismo flujo).
- Sin dependencias externas.

### Compatibilidad
No cambian rutas ni formato esperado desde frontend:
- `Content-Type: multipart/form-data; boundary=...`
- Campo de archivo: `file`

## Benchmark automático de modelos Ollama

Permite comparar modelos de chat detectados en Ollama y guardar histórico para análisis.

### Endpoints
- `POST /api/models/benchmark`
- `GET /api/models/benchmark/results`

### Métricas comparativas
Por cada modelo probado se registran:
- `avg_latency_ms`: latencia promedio por prompt.
- `avg_quality_score`: calidad estimada (heurística por cobertura + longitud + estructura).
- `memory_rss_mb`: uso de memoria RSS del proceso durante benchmark.

### Almacenamiento histórico
- Archivo persistente: `chroma_db/model_benchmark_results.json`.
- Se guarda cada corrida con `run_id`, prompts usados, modelos evaluados y resultados.

### Ejemplo de resultado JSON
```json
{
  "run_id": 1730000000000,
  "started_at_ms": 1730000000000,
  "finished_at_ms": 1730000002500,
  "prompts": [
    "Explica brevemente qué es inteligencia artificial y menciona dos riesgos."
  ],
  "models_tested": ["llama3.1:8b", "mistral:7b"],
  "results": [
    {
      "model": "llama3.1:8b",
      "avg_latency_ms": 420.5,
      "avg_quality_score": 0.73,
      "memory_rss_mb": 312.4,
      "memory_source": "resource",
      "ok_prompts": 1,
      "failed_prompts": 0,
      "prompt_results": [
        {
          "prompt": "Explica brevemente qué es inteligencia artificial y menciona dos riesgos.",
          "latency_ms": 420.5,
          "quality": {
            "score": 0.73,
            "coverage": 0.5,
            "length_chars": 388,
            "structure_score": 1.0
          }
        }
      ]
    }
  ]
}
```

### Ejecución rápida
```bash
curl -sS -X POST http://127.0.0.1:7860/api/models/benchmark \
  -H 'Content-Type: application/json' \
  -d '{"limit_models": 3}'

curl -sS 'http://127.0.0.1:7860/api/models/benchmark/results?limit=5'
```

## Despliegue Docker (backend + UI + Ollama)

Esta configuración **no reemplaza** ni modifica el arranque local con `run_rag_ui_windows.bat`; es una vía adicional para ejecutar en otros equipos.

### Archivos añadidos
- `Dockerfile`: imagen Python para backend RAG/UI (`web_ui.py`).
- `docker-compose.yml`: orquesta `backend`, `ui` (Nginx reverse proxy) y `ollama`.
- `docker/nginx.conf`: proxy de UI/API hacia `backend:7860`.

### Arquitectura Docker
- `backend`:
  - ejecuta `python web_ui.py` (sirve HTML + API).
  - persiste datos en volumen `rag_data` (`/app/data`) y vector DB en `rag_chroma` (`/app/chroma_db`).
  - usa Ollama remoto vía `OLLAMA_HOST=http://ollama:11434`.
- `ui`:
  - Nginx expone `http://localhost:7860` y reenvía al backend.
- `ollama`:
  - servidor local de modelos (`11434`) con volumen persistente `ollama_data`.

### Uso rápido
```bash
docker compose up -d --build
```

Abrir:
- UI: `http://127.0.0.1:7860`
- Ollama API: `http://127.0.0.1:11434`

### Verificar persistencia
1. Subir un archivo desde UI o por API (`/api/upload-file`).
2. Ejecutar indexado.
3. Reiniciar stack:
```bash
docker compose down
docker compose up -d
```
4. Validar que los datos siguen presentes (`/api/status` muestra inventario y estado).

### Pull de modelos en Ollama (contenedor)
```bash
docker exec -it rag_ollama ollama pull nomic-embed-text
docker exec -it rag_ollama ollama pull llama3.1:8b
```

### Comandos útiles
```bash
docker compose logs -f backend
docker compose logs -f ui
docker compose logs -f ollama
```

## Auditoría final de sincronía (entrega)

Checklist aplicado para no romper flujo actual:
- El arranque local Windows (`run_rag_ui_windows.bat`) se mantiene intacto.
- El backend Python local (`web_ui.py`, `rag_mejorado.py`) no cambió su contrato de endpoints existente.
- Docker usa la misma app (`web_ui.py`) y mismas rutas API, evitando divergencia funcional.
- Persistencia separada por volúmenes Docker (`rag_data`, `rag_chroma`, `ollama_data`) para conservar documentos, índices y modelos.
- Se validó sintaxis, tests y endpoints clave tras cambios.

## Gestión de dependencias oficial con uv

A partir de esta versión, el proyecto soporta `uv` como gestor principal de dependencias para mejorar reproducibilidad y mantenimiento.

### Estructura final (dependencias)
```text
coxx/
├─ pyproject.toml        # Configuración de proyecto y dependencias (uv)
├─ uv.lock               # Lockfile reproducible (generado por uv lock)
├─ requirements.txt      # Compatibilidad legacy (pip / entornos previos)
├─ run_rag_ui_windows.bat
├─ rag_mejorado.py
├─ web_ui.py
└─ ...
```

### Archivos clave
- `pyproject.toml`: fuente oficial de dependencias.
- `uv.lock`: lock reproducible para instalaciones consistentes.
- `requirements.txt`: se mantiene por compatibilidad con flujo previo.

### Instalación paso a paso con uv
1. Instalar uv:
```bash
pip install uv
```

2. Crear entorno virtual automático con uv (se crea `.venv`):
```bash
uv sync
```

3. Verificar entorno virtual generado:
```bash
uv run python -c "import sys, pathlib; print(sys.executable); print(pathlib.Path('.venv').exists())"
```

4. Ejecutar aplicación:
```bash
uv run web_ui.py
```

5. Ejecutar pruebas unitarias:
```bash
uv run python -m unittest discover -s tests -v
```

### Generar / regenerar lock reproducible
```bash
uv lock
```

> Nota: si no tienes red en el entorno actual, `uv lock` puede fallar temporalmente; vuelve a ejecutarlo en un entorno con acceso a índice de paquetes.

### Compatibilidad con arranque actual
- El launcher `run_rag_ui_windows.bat` **no se rompe**:
  - si detecta `uv` + `pyproject.toml`, usa `uv sync`/`uv run`.
  - si no, cae automáticamente a `pip install -r requirements.txt`.

## Configuración centralizada (.env + ConfigManager)

Se centralizó la configuración en:
- `.env` (variables editables por entorno)
- `config_manager.py` (lectura tipada, defaults, validaciones y recarga)

### Ejemplo `.env`
```env
RAG_CHAT_MODEL=llama3.1:8b
RAG_EMBED_MODEL=nomic-embed-text
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4
RAG_ENABLE_CONTEXT_COMPRESSION=0
RAG_SEMANTIC_CACHE_ENABLED=1
UI_HOST=0.0.0.0
UI_PORT=7860
RAG_DATA_DIR=./data
RAG_CHROMA_DIR=./chroma_db
```

### Qué valida ConfigManager
- enteros y booleanos tipados,
- rangos (`UI_PORT`, `RAG_TOP_K`, etc.),
- coherencia (`RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE`).

### Recarga dinámica
- Backend expone `POST /api/config/reload` para recargar `.env` en runtime.
- También disponible en código: `rag_mejorado.reload_config()`.

### Compatibilidad
- No se rompe ejecución actual: `.bat`, `requirements.txt`, `uv` y endpoints existentes siguen funcionando.

## Logging estructurado profesional

Backend usa logging JSON estructurado con salida dual:
- Consola (stdout)
- Archivo rotativo

Implementación: `structured_logging.py`.

### Variables `.env` relevantes
```env
RAG_LOG_LEVEL=INFO
RAG_LOG_FILE=./logs/rag_backend.log
RAG_LOG_MAX_BYTES=1048576
RAG_LOG_BACKUP_COUNT=5
```

### Niveles soportados
- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

### Eventos registrados
- eventos de sistema (arranque, indexado, recarga config),
- errores y excepciones,
- uso de modelos,
- métricas operativas,
- acciones de usuario HTTP (GET/POST/DELETE en API UI/backend).

### Ejemplo de log JSON
```json
{
  "ts_ms": 1770878475765,
  "level": "INFO",
  "logger": "web_ui",
  "event": "HTTP POST request",
  "module": "web_ui",
  "line": 930,
  "path": "/api/ask",
  "client": "127.0.0.1"
}
```
