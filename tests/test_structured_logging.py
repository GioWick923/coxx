import json
import os
import tempfile
import unittest
from pathlib import Path

from structured_logging import StructuredJsonFormatter, get_structured_logger


class StructuredLoggingTests(unittest.TestCase):
    def test_json_formatter_includes_event_and_extra(self):
        import logging

        formatter = StructuredJsonFormatter()
        rec = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="evento",
            args=(),
            exc_info=None,
        )
        rec.user_action = "ask"
        payload = json.loads(formatter.format(rec))
        self.assertEqual(payload["event"], "evento")
        self.assertEqual(payload["user_action"], "ask")
        self.assertEqual(payload["level"], "INFO")

    def test_logger_writes_rotating_file(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "backend.log"
            old_file = os.environ.get("RAG_LOG_FILE")
            old_level = os.environ.get("RAG_LOG_LEVEL")
            os.environ["RAG_LOG_FILE"] = str(log_path)
            os.environ["RAG_LOG_LEVEL"] = "DEBUG"
            try:
                logger = get_structured_logger("test_structured_logging")
                logger.debug("debug event", extra={"metric": 1})
                # flush handlers
                for h in logger.handlers:
                    h.flush()
                self.assertTrue(log_path.exists())
                content = log_path.read_text(encoding="utf-8")
                self.assertIn('"event": "debug event"', content)
                self.assertIn('"metric": 1', content)
            finally:
                if old_file is None:
                    os.environ.pop("RAG_LOG_FILE", None)
                else:
                    os.environ["RAG_LOG_FILE"] = old_file
                if old_level is None:
                    os.environ.pop("RAG_LOG_LEVEL", None)
                else:
                    os.environ["RAG_LOG_LEVEL"] = old_level


if __name__ == "__main__":
    unittest.main()
