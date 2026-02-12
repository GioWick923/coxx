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
