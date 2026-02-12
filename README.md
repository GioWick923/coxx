# RAG Studio Pro (Local)

Proyecto RAG local con:
- `rag_mejorado.py` (pipeline RAG incremental + metadata enriquecida)
- `web_ui.py` (UI local con auditoría y carga de fuentes)
- `run_rag_ui_windows.bat` (arranque de un clic en Windows)

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
