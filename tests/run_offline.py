"""Run isolated regression tests without provider keys or existing user data."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    os.environ["LLM_API_KEY"] = ""
    os.environ["LLM_DISABLED"] = "false"
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hashing"
    os.environ["MEMORY_EMBEDDING_PROVIDER"] = "hashing"
    with tempfile.TemporaryDirectory() as folder:
        os.environ["CASEFORGE_DATA_DIR"] = folder
        result = unittest.TextTestRunner(verbosity=1).run(
            unittest.defaultTestLoader.discover(str(ROOT / "tests")))
        raise SystemExit(not result.wasSuccessful())
