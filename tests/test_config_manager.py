import os
import tempfile
import unittest
from pathlib import Path

from config_manager import ConfigManager, ConfigValidationError


class ConfigManagerTests(unittest.TestCase):
    def test_load_defaults_when_env_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            cm = ConfigManager(Path(td) / ".env.missing")
            cfg = cm.get()
            self.assertEqual(cfg.rag_chat_model, "llama3.1:8b")
            self.assertEqual(cfg.ui_port, 7860)

    def test_load_from_env_file_and_typed_values(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "RAG_CHAT_MODEL=mistral:7b\n"
                "RAG_CHUNK_SIZE=900\n"
                "RAG_ENABLE_CONTEXT_COMPRESSION=1\n"
                "UI_PORT=9001\n"
                "RAG_LOG_LEVEL=DEBUG\n"
                "RAG_LOG_MAX_BYTES=2048\n",
                encoding="utf-8",
            )
            old_level = os.environ.pop("RAG_LOG_LEVEL", None)
            old_max = os.environ.pop("RAG_LOG_MAX_BYTES", None)
            try:
                cm = ConfigManager(env_path)
                cfg = cm.get()
                self.assertEqual(cfg.rag_chat_model, "mistral:7b")
                self.assertEqual(cfg.rag_chunk_size, 900)
                self.assertTrue(cfg.rag_enable_context_compression)
                self.assertEqual(cfg.ui_port, 9001)
                self.assertEqual(cfg.rag_log_level, "DEBUG")
                self.assertEqual(cfg.rag_log_max_bytes, 2048)
            finally:
                if old_level is not None:
                    os.environ["RAG_LOG_LEVEL"] = old_level
                if old_max is not None:
                    os.environ["RAG_LOG_MAX_BYTES"] = old_max

    def test_reload_dynamic(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("UI_PORT=7860\n", encoding="utf-8")
            cm = ConfigManager(env_path)
            self.assertEqual(cm.get().ui_port, 7860)

            env_path.write_text("UI_PORT=8001\n", encoding="utf-8")
            cm.reload()
            self.assertEqual(cm.get().ui_port, 8001)

    def test_invalid_values_raise(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("RAG_CHUNK_SIZE=-5\n", encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                ConfigManager(env_path)

    def test_os_env_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("RAG_CHAT_MODEL=foo:1b\n", encoding="utf-8")
            old = os.environ.get("RAG_CHAT_MODEL")
            os.environ["RAG_CHAT_MODEL"] = "bar:2b"
            try:
                cm = ConfigManager(env_path)
                self.assertEqual(cm.get().rag_chat_model, "bar:2b")
            finally:
                if old is None:
                    os.environ.pop("RAG_CHAT_MODEL", None)
                else:
                    os.environ["RAG_CHAT_MODEL"] = old


if __name__ == "__main__":
    unittest.main()
