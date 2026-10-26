import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from common.contracts import Model
from common.models import check_snapshot, model_id, prepare_model
from common.quick import run_quick
from main import app


def make_snapshot(tmp_path, sharded=False):
    snapshot = tmp_path / "models--HuggingFaceTB--SmolLM3-3B" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}")
    if sharded:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
        )
        for part in ("model-1.safetensors", "model-2.safetensors"):
            (snapshot / part).write_bytes(b"unit-test-placeholder")
    else:
        (snapshot / "model.safetensors").write_bytes(b"unit-test-placeholder")
    return snapshot


@pytest.mark.parametrize("flag", ["--model", "-model", "-m"])
def test_quick_cli_accepts_requested_syntax_without_config(monkeypatch, flag):
    calls = []
    monkeypatch.setattr("common.quick.run_quick", lambda *args: calls.append(args) or "saved")
    result = CliRunner().invoke(app, ["-n", "5", flag, "SmolLM3"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0:3] == (5, "SmolLM3", Path("/models"))
    assert "saved" in result.output


def test_unknown_alias_does_not_download(monkeypatch):
    monkeypatch.setattr("common.models._snapshot_download", lambda **kw: pytest.fail("download"))
    with pytest.raises(ValueError, match="desconocido"):
        prepare_model("not-a-model")
    assert model_id("sMoLlM3") == "HuggingFaceTB/SmolLM3-3B"


def test_existing_snapshot_is_reused_without_network(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"]
        return str(snapshot)

    monkeypatch.setattr("common.models._snapshot_download", download)
    model = prepare_model("SmolLM3", tmp_path)
    assert model.path == str(snapshot)
    assert model.revision == "a" * 40
    assert len(calls) == 1
    assert calls[0]["cache_dir"] == str(tmp_path)


def test_missing_snapshot_downloads_once_then_pins_revision(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if kwargs["local_files_only"]:
            raise FileNotFoundError("cache missing")
        return str(snapshot)

    monkeypatch.setattr("common.models._snapshot_download", download)
    model = prepare_model("SmolLM3", tmp_path, revision="a" * 40)
    assert [c["local_files_only"] for c in calls] == [True, False]
    assert all(c["revision"] == "a" * 40 for c in calls)
    assert model.revision == "a" * 40


def test_offline_rejects_missing_shard_without_download(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path, sharded=True)
    (snapshot / "model-2.safetensors").unlink()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("common.models._snapshot_download", download)
    with pytest.raises(ValueError, match="snapshot completo"):
        prepare_model("SmolLM3", tmp_path, offline=True)
    assert len(calls) == 1 and calls[0]["local_files_only"]
    with pytest.raises(ValueError, match="incompleto"):
        check_snapshot(snapshot)


def test_quick_runs_only_base_with_exact_limit_and_one_repetition(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parent.parent
    calls = []
    monkeypatch.setattr(
        "common.quick.prepare_model",
        lambda *args, **kwargs: Model(
            id="HuggingFaceTB/SmolLM3-3B", path=str(tmp_path), revision="a" * 40
        ),
    )
    monkeypatch.setattr("common.generation.infer", lambda *args: calls.append(args) or tmp_path)
    assert (
        run_quick(3, requests=root / "examples/smoke/requests.jsonl", output=tmp_path) == tmp_path
    )
    cfg, module, shots = calls[0]
    assert module == "base_model" and shots == 0
    assert cfg.phase == "development" and cfg.request_limit == 3 and cfg.repetitions == 1
    assert cfg.chat_template_kwargs == {"enable_thinking": False}


def test_excessive_count_never_downloads_or_repeats_inputs(monkeypatch):
    monkeypatch.setattr("common.quick.prepare_model", lambda *a, **kw: pytest.fail("download"))
    with pytest.raises(ValueError, match="contiene 5 solicitudes"):
        run_quick(6)


def test_quick_options_not_silently_ignored_with_subcommands():
    result = CliRunner().invoke(app, ["-n", "2", "summarize", "--source", "missing"])
    assert result.exit_code != 0
    assert "sin subcomando" in result.output
