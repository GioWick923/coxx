import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment_validator import EnvironmentValidator, get_environment_status


class EnvironmentValidatorTests(unittest.TestCase):
    def test_validate_dependencies_reports_missing_modules(self):
        validator = EnvironmentValidator()
        with patch.object(validator, "REQUIRED_MODULES", ["json", "nonexistent_mod_xyz"]):
            result = validator.validate_dependencies()
        self.assertFalse(result.ok)
        self.assertIn("nonexistent_mod_xyz", result.details["missing_modules"])

    def test_validate_ollama_missing_binary(self):
        validator = EnvironmentValidator(required_models=["a", "b"])
        with patch("environment_validator.shutil.which", return_value=None):
            result = validator.validate_ollama()
        self.assertFalse(result.ok)
        self.assertTrue(any("ollama" in err.lower() for err in result.errors))

    def test_validate_ollama_detects_models_and_missing_required(self):
        validator = EnvironmentValidator(required_models=["llama3.1:8b", "nomic-embed-text"])

        class Proc:
            returncode = 0
            stdout = "NAME ID SIZE MODIFIED\nllama3.1:8b abc 1GB now\n"
            stderr = ""

        with patch("environment_validator.shutil.which", return_value="/usr/bin/ollama"), patch(
            "environment_validator.subprocess.run", return_value=Proc()
        ):
            result = validator.validate_ollama()

        self.assertFalse(result.ok)
        self.assertIn("llama3.1:8b", result.details["detected_models"])
        self.assertIn("nomic-embed-text", result.details["missing_required_models"])

    def test_validate_disk_access_success(self):
        validator = EnvironmentValidator()
        with tempfile.TemporaryDirectory() as td:
            validator.storage_paths = [Path(td) / "data", Path(td) / "db", Path(td) / "logs"]
            result = validator.validate_disk_access()
        self.assertTrue(result.ok)
        self.assertEqual(len(result.details["writable_paths"]), 3)

    def test_validate_aggregate_shape(self):
        validator = EnvironmentValidator()
        fake_ok = {"ok": True, "errors": [], "warnings": [], "suggestions": [], "details": {}}
        fake_warn = {
            "ok": False,
            "errors": ["error"],
            "warnings": ["warn"],
            "suggestions": ["fix it"],
            "details": {},
        }
        with patch.object(validator, "validate_python") as p, patch.object(
            validator, "validate_dependencies"
        ) as d, patch.object(validator, "validate_ollama") as o, patch.object(
            validator, "validate_disk_access"
        ) as k:
            p.return_value = type("R", (), fake_ok)()
            d.return_value = type("R", (), fake_ok)()
            o.return_value = type("R", (), fake_warn)()
            k.return_value = type("R", (), fake_ok)()
            out = validator.validate()

        self.assertIn("checks", out)
        self.assertIn("ollama", out["checks"])
        self.assertFalse(out["ok"])
        self.assertIn("error", out["errors"])


class EnvironmentValidatorHelperTests(unittest.TestCase):
    def test_get_environment_status_returns_expected_keys(self):
        out = get_environment_status()
        self.assertIn("ok", out)
        self.assertIn("checks", out)
        self.assertIn("timestamp", out)


if __name__ == "__main__":
    unittest.main()
