import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import rag_mejorado as rag


class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SemanticCacheTests(unittest.TestCase):
    def test_cache_put_get(self):
        with tempfile.TemporaryDirectory() as td:
            cache = rag.SemanticCache(Path(td) / "cache.json")
            cache.put("Hola mundo", "respuesta")
            self.assertEqual(cache.get("hola   mundo"), "respuesta")


class MetadataTests(unittest.TestCase):
    def test_enrich_metadata_contains_required_fields(self):
        d = DummyDoc("contenido", {"page": 2})
        m = rag._enrich_metadata(d, "a.txt", "txt")
        self.assertIn("source", m)
        self.assertIn("file_type", m)
        self.assertIn("page_number", m)
        self.assertIn("original_document_id", m)
        self.assertIn("embedding_version", m)
        self.assertEqual(m["file_type"], "txt")


class ContextBuilderTests(unittest.TestCase):
    def test_context_deduplicates(self):
        docs = [
            DummyDoc("mismo contenido", {"source": "s1", "file_type": "txt", "page_number": 1}),
            DummyDoc("mismo contenido", {"source": "s2", "file_type": "txt", "page_number": 2}),
        ]
        context = rag._build_context(docs)
        self.assertEqual(context.count("mismo contenido"), 1)


class RagChatTests(unittest.TestCase):
    def test_rag_chat_uses_cache(self):
        vectorstore = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            with patch.object(rag, "SEMANTIC_CACHE_ENABLED", True), patch.object(
                rag, "SEMANTIC_CACHE_FILE", Path(td) / "semantic_cache.json"
            ), patch.object(rag.ModelRegistry, "llm") as llm:
                llm.return_value.invoke.return_value = "{\"answer\":\"ok\"}"
                vectorstore.similarity_search_with_relevance_scores.return_value = [
                    (DummyDoc("c1", {"source": "s", "file_type": "txt", "page_number": None}), 0.9)
                ]
                first = rag.rag_chat("q1", vectorstore)
                second = rag.rag_chat("q1", vectorstore)
                self.assertEqual(first, second)


class RegistryTests(unittest.TestCase):
    def test_embedding_version_stable(self):
        v1 = rag._embedding_version()
        v2 = rag._embedding_version()
        self.assertEqual(v1, v2)
        self.assertTrue(len(v1) > 5)


class WebSourceTests(unittest.TestCase):
    def test_add_and_list_web_source(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(rag, "DATA_FOLDER", Path(td)), patch.object(
                rag, "WEB_SOURCES_FILE", Path(td) / "web_sources.txt"
            ), patch.object(rag, "WIKI_URL", "https://base.local"):
                rag.add_web_source("https://example.com/doc")
                urls = rag.list_web_sources()
                self.assertIn("https://base.local", urls)
                self.assertIn("https://example.com/doc", urls)


class IncrementalIndexTests(unittest.TestCase):
    def test_classify_changes_by_source_stable(self):
        docs = [
            DummyDoc("p1", {"source": "a.pdf", "file_type": "pdf", "source_id": "s1", "original_document_id": "h1"}),
            DummyDoc("p2", {"source": "a.pdf", "file_type": "pdf", "source_id": "s1", "original_document_id": "h2"}),
        ]
        registry = {"sources": {}}
        changed, touched, stats = rag._classify_changes(docs, registry)
        self.assertEqual(len(changed), 2)
        self.assertEqual(stats["new_docs"], 1)
        self.assertEqual(len(touched), 1)




class ModelSelectionTests(unittest.TestCase):
    def test_suggest_models_prefers_embed_and_chat(self):
        models = ["nomic-embed-text", "llama3.2:latest"]
        suggested = rag.suggest_models(models)
        self.assertEqual(suggested["embed_model"], "nomic-embed-text")
        self.assertEqual(suggested["chat_model"], "llama3.2:latest")

    def test_set_active_models_auto_detect(self):
        with patch.object(rag, "list_ollama_models", return_value=["nomic-embed-text", "mistral:7b"]):
            out = rag.set_active_models(auto_detect=True)
            self.assertEqual(out["embed_model"], "nomic-embed-text")
            self.assertEqual(out["chat_model"], "mistral:7b")


class ModelTaskRoutingTests(unittest.TestCase):
    def test_classify_models_by_task_type(self):
        models = ["nomic-embed-text", "llama3.1:8b", "deepseek-r1:32b"]
        grouped = rag.classify_models(models)
        self.assertIn("nomic-embed-text", grouped[rag.TASK_EMBEDDING])
        self.assertIn("llama3.1:8b", grouped[rag.TASK_CHAT])
        self.assertIn("deepseek-r1:32b", grouped[rag.TASK_DEEP_ANALYSIS])

    def test_select_best_model_for_each_task(self):
        models = ["nomic-embed-text", "llama3.1:8b", "deepseek-r1:32b"]
        self.assertEqual(
            rag.select_best_model_for_task(rag.TASK_EMBEDDING, models),
            "nomic-embed-text",
        )
        self.assertEqual(
            rag.select_best_model_for_task(rag.TASK_CHAT, models),
            "llama3.1:8b",
        )
        self.assertEqual(
            rag.select_best_model_for_task(rag.TASK_DEEP_ANALYSIS, models),
            "deepseek-r1:32b",
        )

    def test_select_models_for_tasks_compact_api(self):
        models = ["nomic-embed-text", "llama3.1:8b", "deepseek-r1:32b"]
        selected = rag.select_models_for_tasks(installed_models=models)
        self.assertEqual(selected["embed_model"], "nomic-embed-text")
        self.assertEqual(selected["chat_model"], "llama3.1:8b")
        self.assertEqual(selected["analysis_model"], "deepseek-r1:32b")


class ModelActivationTests(unittest.TestCase):
    """Pruebas de detección y aplicación de modelos activos."""

    def test_detect_installed_models_from_ollama_output(self):
        """1) Detecta correctamente modelos instalados parseando `ollama list`."""
        from subprocess import CompletedProcess

        fake_stdout = (
            "NAME ID SIZE MODIFIED\n"
            "llama3.1:8b abc 4.7 GB 2 days ago\n"
            "nomic-embed-text def 274 MB 3 days ago\n"
        )
        with patch("rag_mejorado.subprocess.run", return_value=CompletedProcess(args=["ollama", "list"], returncode=0, stdout=fake_stdout, stderr="")):
            models = rag.list_ollama_models()

        self.assertEqual(models, ["llama3.1:8b", "nomic-embed-text"])

    def test_apply_active_chat_model_success(self):
        """2) Aplica correctamente el modelo activo de chat."""
        with patch.object(rag, "list_ollama_models", return_value=["llama3.1:8b", "nomic-embed-text"]):
            out = rag.set_active_models(chat_model="llama3.1:8b")
        self.assertEqual(out["chat_model"], "llama3.1:8b")

    def test_apply_active_embedding_model_success(self):
        """3) Aplica correctamente el modelo activo de embeddings."""
        with patch.object(rag, "list_ollama_models", return_value=["llama3.1:8b", "nomic-embed-text"]):
            out = rag.set_active_models(embed_model="nomic-embed-text")
        self.assertEqual(out["embed_model"], "nomic-embed-text")

    def test_set_active_models_fails_when_model_not_installed(self):
        """4) Falla de forma controlada cuando se solicita un modelo no instalado."""
        with patch.object(rag, "list_ollama_models", return_value=["llama3.1:8b", "nomic-embed-text"]):
            with self.assertRaises(ValueError):
                rag.set_active_models(chat_model="modelo-inexistente:1b")
            with self.assertRaises(ValueError):
                rag.set_active_models(embed_model="otro-inexistente")


class ModelMetricsExposureTests(unittest.TestCase):
    def test_get_model_metrics_includes_detected_active_latency_memory(self):
        tracker = rag.ModelMetricsTracker()
        tracker.record(rag.TASK_CHAT, "llama3.1:8b", 120)
        tracker.record(rag.TASK_CHAT, "llama3.1:8b", 80)
        tracker.record(rag.TASK_EMBEDDING, "nomic-embed-text", 40)

        with patch.object(rag, "MODEL_METRICS", tracker), patch.object(
            rag,
            "list_ollama_models",
            return_value=["llama3.1:8b", "nomic-embed-text"],
        ), patch.object(rag, "ACTIVE_CHAT_MODEL", "llama3.1:8b"), patch.object(
            rag,
            "ACTIVE_EMBED_MODEL",
            "nomic-embed-text",
        ):
            payload = rag.get_model_metrics()

        self.assertIn("active_models", payload)
        self.assertIn("detected_models", payload)
        self.assertIn("performance", payload)
        self.assertIn("memory", payload)
        self.assertEqual(payload["performance"]["chat"]["avg_latency_ms"], 100.0)
        self.assertEqual(payload["performance"]["embedding"]["avg_latency_ms"], 40.0)
        self.assertEqual(payload["active_models"]["chat"]["name"], "llama3.1:8b")


class AdaptivePromptTemplateTests(unittest.TestCase):
    def test_choose_summary_template(self):
        key = rag.choose_prompt_template(
            "Haz un resumen ejecutivo de este documento",
            "contexto breve",
        )
        self.assertEqual(key, "summary_long")

    def test_choose_reasoning_template(self):
        key = rag.choose_prompt_template(
            "Analiza paso a paso las diferencias",
            "contexto breve",
        )
        self.assertEqual(key, "reasoning_steps")

    def test_choose_qa_short_vs_long(self):
        short_key = rag.choose_prompt_template("¿Qué dice el texto?", "abc")
        long_key = rag.choose_prompt_template("¿Qué dice el texto?", "x" * 3000)
        self.assertEqual(short_key, "qa_short")
        self.assertEqual(long_key, "qa_long")

    def test_build_prompt_uses_selected_template(self):
        prompt = rag._build_prompt("Haz un resumen", "contenido")
        self.assertIn('"summary"', prompt)
        prompt2 = rag._build_prompt("Analiza paso a paso", "contenido")
        self.assertIn('"steps"', prompt2)


class ModelProfilesTests(unittest.TestCase):
    def test_upsert_and_list_profiles(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(rag, "CHROMA_DIR", Path(td)), patch.object(
                rag, "MODEL_PROFILES_FILE", Path(td) / "model_profiles.json"
            ):
                saved = rag.upsert_model_profile(
                    name="MiPerfil",
                    chat_model="llama3.1:8b",
                    embed_model="nomic-embed-text",
                    description="perfil custom",
                    generation={"temperature": 0.3, "top_p": 0.8},
                    rag={"top_k": 6, "max_chars_per_chunk": 888, "max_total_context_chars": 3999},
                )
                self.assertEqual(saved["name"], "MiPerfil")
                self.assertEqual(saved["generation"]["temperature"], 0.3)
                profiles = rag.list_model_profiles()
                self.assertIn("MiPerfil", profiles)

    def test_apply_profile_updates_active_models(self):
        profiles = {
            "Rapido": {
                "description": "x",
                "chat_model": "llama3.1:8b",
                "embed_model": "nomic-embed-text",
            }
        }
        with patch.object(rag, "_load_model_profiles", return_value=profiles), patch.object(
            rag,
            "set_active_models",
            return_value={
                "chat_model": "llama3.1:8b",
                "embed_model": "nomic-embed-text",
                "installed_models": ["llama3.1:8b", "nomic-embed-text"],
            },
        ):
            out = rag.apply_model_profile("Rapido")
            self.assertEqual(out["profile"], "Rapido")
            self.assertEqual(out["chat_model"], "llama3.1:8b")
            self.assertIn("generation", out)
            self.assertIn("rag", out)

    def test_delete_profile(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(rag, "CHROMA_DIR", Path(td)), patch.object(
                rag, "MODEL_PROFILES_FILE", Path(td) / "model_profiles.json"
            ):
                rag.upsert_model_profile(
                    name="Borrar",
                    chat_model="llama3.1:8b",
                    embed_model="nomic-embed-text",
                )
                self.assertTrue(rag.delete_model_profile("Borrar"))
                profiles = rag.list_model_profiles()
                self.assertNotIn("Borrar", profiles)


class ConfigurationExportImportTests(unittest.TestCase):
    def test_export_configuration_contains_required_sections(self):
        with patch.object(rag, "list_model_profiles", return_value={"A": {"chat_model": "c", "embed_model": "e", "generation": {}, "rag": {}, "description": "d"}}), patch.object(
            rag,
            "get_runtime_config",
            return_value={"x": 1},
        ), patch.object(rag, "ACTIVE_CHAT_MODEL", "c"), patch.object(rag, "ACTIVE_EMBED_MODEL", "e"):
            out = rag.export_configuration()
        self.assertIn("schema_version", out)
        self.assertIn("profiles", out)
        self.assertIn("active", out)
        self.assertIn("runtime", out)

    def test_import_configuration_conflict_requires_overwrite(self):
        payload = {
            "schema_version": rag.CONFIG_EXPORT_VERSION,
            "profiles": {
                "Rapido": {
                    "description": "d",
                    "chat_model": "llama3.1:8b",
                    "embed_model": "nomic-embed-text",
                    "generation": {"temperature": 0.1, "top_p": 0.9},
                    "rag": {"top_k": 4, "max_chars_per_chunk": 1000, "max_total_context_chars": 3500},
                }
            },
            "active": {
                "chat_model": "llama3.1:8b",
                "embed_model": "nomic-embed-text",
                "generation": {"temperature": 0.1, "top_p": 0.9},
                "rag": {"top_k": 4, "max_chars_per_chunk": 1000, "max_total_context_chars": 3500},
            },
        }
        with patch.object(rag, "list_ollama_models", return_value=["llama3.1:8b", "nomic-embed-text"]), patch.object(
            rag,
            "list_model_profiles",
            return_value={"Rapido": {"chat_model": "llama3.1:8b", "embed_model": "nomic-embed-text"}},
        ):
            with self.assertRaises(ValueError):
                rag.import_configuration(payload, overwrite_profiles=False)

    def test_import_configuration_success_with_overwrite(self):
        payload = {
            "schema_version": rag.CONFIG_EXPORT_VERSION,
            "profiles": {
                "Nuevo": {
                    "description": "d",
                    "chat_model": "llama3.1:8b",
                    "embed_model": "nomic-embed-text",
                    "generation": {"temperature": 0.2, "top_p": 0.8},
                    "rag": {"top_k": 4, "max_chars_per_chunk": 1000, "max_total_context_chars": 3500},
                }
            },
            "active": {
                "chat_model": "llama3.1:8b",
                "embed_model": "nomic-embed-text",
                "generation": {"temperature": 0.2, "top_p": 0.8},
                "rag": {"top_k": 4, "max_chars_per_chunk": 1000, "max_total_context_chars": 3500},
            },
        }
        with patch.object(rag, "list_ollama_models", return_value=["llama3.1:8b", "nomic-embed-text"]), patch.object(
            rag,
            "list_model_profiles",
            return_value={"Rapido": {"chat_model": "llama3.1:8b", "embed_model": "nomic-embed-text"}},
        ), patch.object(rag, "_save_model_profiles") as save_mock, patch.object(
            rag,
            "set_active_models",
            return_value={"chat_model": "llama3.1:8b", "embed_model": "nomic-embed-text", "installed_models": []},
        ):
            out = rag.import_configuration(payload, overwrite_profiles=True)
        self.assertIn("imported_profiles", out)
        self.assertTrue(save_mock.called)


if __name__ == "__main__":
    unittest.main()
