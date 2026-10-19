"""Check portable model discovery without CUDA or development-machine paths."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from perceive import _find_nano


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_snapshot_works_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "downloaded_model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps({"model_type": "cosmos3"})
            )
            with patch.dict(
                os.environ, {"COSMOS3_NANO_PATH": str(checkpoint)}, clear=True
            ):
                self.assertEqual(_find_nano(), checkpoint)

    def test_shared_hub_cache_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "shared_cache"
            checkpoint = hub / "models--nvidia--Cosmos3-Nano/snapshots/revision"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True):
                self.assertEqual(_find_nano(), checkpoint)

    def test_incomplete_download_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"COSMOS3_NANO_PATH": directory}, clear=True):
                with self.assertRaisesRegex(SystemExit, "missing config.json"):
                    _find_nano()


if __name__ == "__main__":
    unittest.main()
