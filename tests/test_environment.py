import os
import sys
from types import ModuleType

from typer.testing import CliRunner

from common.environment import load_repo_env
from common.inference import offline_environment
from main import app


def test_dotenv_reads_quoted_token_without_printing_or_interpolation(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".env"
    path.write_text('# comment\nexport HF_TOKEN="test-placeholder-${SOMETHING}" # comment\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("SOMETHING", "must-not-be-expanded")
    load_repo_env(path)
    assert os.environ["HF_TOKEN"] == "test-placeholder-${SOMETHING}"
    assert capsys.readouterr().out == ""


def test_dotenv_preserves_shell_token_and_accepts_missing_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=file-placeholder\n")
    monkeypatch.setenv("HF_TOKEN", "shell-placeholder")
    load_repo_env(path)
    assert os.environ["HF_TOKEN"] == "shell-placeholder"
    load_repo_env(tmp_path / "does-not-exist")
    assert os.environ["HF_TOKEN"] == "shell-placeholder"


def test_cli_loads_env_before_model_preparation_without_exposing_token(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=cli-test-placeholder\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("main.load_repo_env", lambda: load_repo_env(path))
    captured = []

    def quick(*args):
        captured.append(os.environ.get("HF_TOKEN"))
        return tmp_path

    monkeypatch.setattr("common.quick.run_quick", quick)
    result = CliRunner().invoke(app, ["-n", "5", "--model", "SmolLM3"])
    assert result.exit_code == 0, result.output
    assert captured == ["cli-test-placeholder"]
    assert "cli-test-placeholder" not in result.output


def test_downloader_reads_environment_without_login_or_token_argument(tmp_path, monkeypatch):
    from common.models import _snapshot_download

    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=download-test-placeholder\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("common.environment.load_repo_env", lambda: load_repo_env(path))
    fake = ModuleType("huggingface_hub")
    captured = []

    def download(**kwargs):
        captured.append((os.environ.get("HF_TOKEN"), kwargs))
        return "snapshot"

    fake.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert _snapshot_download(repo_id="example/model", local_files_only=True) == "snapshot"
    assert captured == [
        ("download-test-placeholder", {"repo_id": "example/model", "local_files_only": True})
    ]


def test_local_inference_children_do_not_receive_hub_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "child-test-placeholder")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy-test-placeholder")
    child = offline_environment()
    assert "HF_TOKEN" not in child and "HUGGING_FACE_HUB_TOKEN" not in child
    assert os.environ["HF_TOKEN"] == "child-test-placeholder"
    assert child["HF_HUB_OFFLINE"] == "1"
