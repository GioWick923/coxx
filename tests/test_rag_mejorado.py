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
        with patch.object(rag, "SEMANTIC_CACHE_ENABLED", True), patch.object(
            rag.ModelRegistry, "llm"
        ) as llm:
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


if __name__ == "__main__":
    unittest.main()
