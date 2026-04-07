"""
Tests for ecgen.f.upload_hf and ecgen.f.upload_hf_single.

Uses unittest.mock to patch huggingface_hub so no real uploads occur
and no HF credentials are required.

Run:
    python tests/test_upload_hf.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make src layout importable when running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecgen.f import upload_hf, upload_hf_single

# Get the actual module object (not the function shadowing it in ecgen.f's namespace)
_MOD = sys.modules["ecgen.f.upload_hf"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_ckpt(tmp_dir: Path, name: str = "best.ckpt") -> Path:
    p = tmp_dir / name
    p.write_bytes(b"fake checkpoint data")
    return p


def _make_fake_dir(tmp_dir: Path, name: str = "checkpoints") -> Path:
    d = tmp_dir / name
    d.mkdir()
    (d / "epoch_001.ckpt").write_bytes(b"epoch 1")
    (d / "epoch_002.ckpt").write_bytes(b"epoch 2")
    return d


# ---------------------------------------------------------------------------
# Dry-run tests — no HF calls, no HF install required
# ---------------------------------------------------------------------------

class TestUploadHfDryRun(unittest.TestCase):

    def test_dry_run_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            config = {
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/ptbxl/best.ckpt"}],
            }
            result = upload_hf(config, dry_run=True)
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})

    def test_dry_run_missing_path_is_skipped(self):
        config = {
            "hf_repo_id": "testuser/ECGEN",
            "checkpoints": [
                {"local_path": "/nonexistent/path/model.ckpt", "repo_path": "vae/model.ckpt"}
            ],
        }
        result = upload_hf(config, dry_run=True)
        self.assertEqual(result["skipped"], 1)

    def test_dry_run_from_yaml_path_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = _make_fake_ckpt(tmp_path)
            import yaml
            config_file = tmp_path / "upload.yaml"
            config_file.write_text(yaml.dump({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/ptbxl/best.ckpt"}],
            }))
            result = upload_hf(config_file, dry_run=True)
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})

    def test_dry_run_from_yaml_str_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = _make_fake_ckpt(tmp_path)
            import yaml
            config_file = tmp_path / "upload.yaml"
            config_file.write_text(yaml.dump({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
            }))
            result = upload_hf(str(config_file), dry_run=True)
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})

    def test_dry_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = _make_fake_dir(Path(tmp))
            config = {
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [{"local_path": str(ckpt_dir), "repo_path": "vae/mimic"}],
            }
            result = upload_hf(config, dry_run=True)
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestUploadHfValidation(unittest.TestCase):

    def test_missing_repo_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            upload_hf({"checkpoints": []})

    def test_empty_checkpoints_returns_zeros(self):
        # Does not require HF installed — returns early before the availability check
        result = upload_hf({"hf_repo_id": "testuser/ECGEN", "checkpoints": []})
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})

    def test_missing_config_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            upload_hf("/nonexistent/config.yaml")

    def test_import_error_when_hf_hub_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            with patch.object(_MOD, "_HF_AVAILABLE", False):
                with self.assertRaises(ImportError):
                    upload_hf({
                        "hf_repo_id": "testuser/ECGEN",
                        "checkpoints": [{"local_path": str(ckpt), "repo_path": "x.ckpt"}],
                    })

    def test_dry_run_works_without_hf_hub(self):
        """dry_run=True must not raise ImportError even if huggingface_hub missing."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            with patch.object(_MOD, "_HF_AVAILABLE", False):
                result = upload_hf(
                    {"hf_repo_id": "testuser/ECGEN",
                     "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}]},
                    dry_run=True,
                )
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})


# ---------------------------------------------------------------------------
# Full upload flow — huggingface_hub mocked out
# ---------------------------------------------------------------------------

class TestUploadHfMocked(unittest.TestCase):
    """All tests here patch HfApi, create_repo, and _HF_AVAILABLE."""

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_single_file_upload(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            result = upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [
                    {"local_path": str(ckpt), "repo_path": "vae/ptbxl/best.ckpt",
                     "description": "VAE best checkpoint"},
                ],
            })

        self.assertEqual(result, {"success": 1, "skipped": 0, "failed": 0})
        mock_create_repo.assert_called_once_with(
            repo_id="testuser/ECGEN", repo_type="model", private=False, exist_ok=True,
            token=None
        )
        mock_api.upload_file.assert_called_once()
        kw = mock_api.upload_file.call_args.kwargs
        self.assertEqual(kw["path_in_repo"], "vae/ptbxl/best.ckpt")
        self.assertEqual(kw["repo_id"], "testuser/ECGEN")

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_directory_upload_uses_upload_folder(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = _make_fake_dir(Path(tmp))
            result = upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [{"local_path": str(ckpt_dir), "repo_path": "vae/mimic"}],
            })

        self.assertEqual(result["success"], 1)
        mock_api.upload_folder.assert_called_once()
        self.assertEqual(mock_api.upload_folder.call_args.kwargs["path_in_repo"], "vae/mimic")
        mock_api.upload_file.assert_not_called()

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_multiple_checkpoints(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt1 = _make_fake_ckpt(tmp_path, "model_a.ckpt")
            ckpt2 = _make_fake_ckpt(tmp_path, "model_b.ckpt")
            result = upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [
                    {"local_path": str(ckpt1), "repo_path": "pulse2pulse/ptbxl/model_a.ckpt"},
                    {"local_path": str(ckpt2), "repo_path": "pulse2pulse/mimic/model_b.ckpt"},
                ],
            })

        self.assertEqual(result, {"success": 2, "skipped": 0, "failed": 0})
        self.assertEqual(mock_api.upload_file.call_count, 2)

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_missing_path_is_skipped_not_failed(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            result = upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "checkpoints": [
                    {"local_path": str(ckpt), "repo_path": "vae/best.ckpt"},
                    {"local_path": "/does/not/exist.ckpt", "repo_path": "vae/missing.ckpt"},
                ],
            })

        self.assertEqual(result, {"success": 1, "skipped": 1, "failed": 0})

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_upload_failure_raises_runtime_error(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value
        mock_api.upload_file.side_effect = Exception("network error")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            with self.assertRaises(RuntimeError):
                upload_hf({
                    "hf_repo_id": "testuser/ECGEN",
                    "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
                })

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_repo_creation_failure_raises_runtime_error(self, MockHfApi, mock_create_repo):
        mock_create_repo.side_effect = Exception("403 Forbidden")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            with self.assertRaises(RuntimeError):
                upload_hf({
                    "hf_repo_id": "testuser/ECGEN",
                    "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
                })

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_private_repo_flag(self, MockHfApi, mock_create_repo):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "private": True,
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
            })
        mock_create_repo.assert_called_once_with(
            repo_id="testuser/ECGEN", repo_type="model", private=True, exist_ok=True,
            token=None
        )

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_custom_commit_message(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "commit_message": "Release v1.0",
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
            })
        commit_msg = mock_api.upload_file.call_args.kwargs["commit_message"]
        self.assertIn("Release v1.0", commit_msg)

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_dataset_repo_type(self, MockHfApi, mock_create_repo):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            upload_hf({
                "hf_repo_id": "testuser/ECGEN",
                "hf_repo_type": "dataset",
                "checkpoints": [{"local_path": str(ckpt), "repo_path": "vae/best.ckpt"}],
            })
        mock_create_repo.assert_called_once_with(
            repo_id="testuser/ECGEN", repo_type="dataset", private=False, exist_ok=True,
            token=None
        )


# ---------------------------------------------------------------------------
# upload_hf_single convenience wrapper
# ---------------------------------------------------------------------------

class TestUploadHfSingle(unittest.TestCase):

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_single_upload(self, MockHfApi, mock_create_repo):
        mock_api = MockHfApi.return_value
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            result = upload_hf_single(
                local_path=str(ckpt),
                repo_path="vae/ptbxl/best.ckpt",
                repo_id="testuser/ECGEN",
            )
        self.assertEqual(result["success"], 1)
        mock_api.upload_file.assert_called_once()

    def test_single_dry_run_no_hf_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            result = upload_hf_single(
                local_path=str(ckpt),
                repo_path="vae/ptbxl/best.ckpt",
                repo_id="testuser/ECGEN",
                dry_run=True,
            )
        self.assertEqual(result, {"success": 0, "skipped": 0, "failed": 0})

    @patch.object(_MOD, "_HF_AVAILABLE", True)
    @patch.object(_MOD, "create_repo")
    @patch.object(_MOD, "HfApi")
    def test_single_passes_repo_type_and_private(self, MockHfApi, mock_create_repo):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _make_fake_ckpt(Path(tmp))
            upload_hf_single(
                local_path=str(ckpt),
                repo_path="vae/best.ckpt",
                repo_id="testuser/ECGEN",
                repo_type="dataset",
                private=True,
            )
        mock_create_repo.assert_called_once_with(
            repo_id="testuser/ECGEN", repo_type="dataset", private=True, exist_ok=True,
            token=None
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestUploadHfDryRun,
        TestUploadHfValidation,
        TestUploadHfMocked,
        TestUploadHfSingle,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
