"""Unit tests for CLI commands and saved results; no model or server is started."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.contracts import Experiment
from common.inference import LocalBench, bench_command, read_detailed, serve_command


def cfg(tmp_path):
    return Experiment(model={"path": str(tmp_path), "revision": "a" * 40}, benchmark="unused")


def native():
    return {
        "generated_texts": ['{"enunciado":"P","solucion":"S"}'],
        "errors": [""],
        "input_lens": [5],
        "output_lens": [3],
        "ttfts": [0.5],
        "itls": [[0.2, 0.3]],
    }


def test_official_commands_are_local_sequential_and_preserve_inputs(tmp_path):
    config = cfg(tmp_path)
    command = serve_command(config, 8000)
    assert command[:2] == ["vllm", "serve"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    command = bench_command(
        config, 8000, tmp_path / "in.jsonl", tmp_path / "out.json", 295, 42, 0.6
    )
    assert command[:3] == ["vllm", "bench", "serve"]
    for flag in ("--skip-chat-template", "--disable-shuffle", "--no-oversample", "--save-detailed"):
        assert flag in command
    assert command[command.index("--max-concurrency") + 1] == "1"
    assert "--ignore-eos" not in command
    body = json.loads(command[command.index("--extra-body") + 1])
    assert body == {"seed": 42, "add_special_tokens": False}


def test_lora_routes_readiness_and_measured_requests_to_same_adapter(tmp_path):
    config = cfg(tmp_path)
    config.adapter = "/local/adapter"
    serve = serve_command(config, 8000, True)
    assert serve[serve.index("--lora-modules") + 1] == "math=/local/adapter"
    bench = bench_command(config, 8000, Path("in"), tmp_path / "out", 1, 42, 0, use_adapter=True)
    assert bench[bench.index("--served-model-name") + 1] == "math"
    assert "--lora-modules" not in bench  # no random adapter selection


def test_metrics_match_thesis_definition_and_retain_limit_warning():
    row = read_detailed(native(), [4], 3)[0]
    assert row["latency_s"] == 1
    assert row["ttft_s"] == 0.5
    assert row["decode_s"] == 0.5
    assert row["decode_tokens_per_second"] == 4
    assert row["tpot_s"] == 0.25
    assert row["usage"]["prompt_tokens"] == 4
    assert row["bench_input_tokens"] == 5
    assert row["output_limit_reached"] is True
    assert row["finish_reason"] is None


def test_incomplete_detailed_results_never_silently_drop_requests():
    with pytest.raises(ValueError, match="incompleto"):
        read_detailed(native(), [4, 4], 10)
    result = native()
    result["errors"] = ["transport failed"]
    row = read_detailed(result, [4], 10)[0]
    assert row["ttft_s"] is None and row["error"]


def test_render_preserves_exact_chat_tokens_and_rejects_overflow(tmp_path):
    engine = LocalBench.__new__(LocalBench)
    engine.cfg = cfg(tmp_path)
    engine.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tokenize, **kw: [1, 2] if tokenize else "rendered",
        encode=lambda text, **kw: [1, 2],
    )
    assert engine.render([])["prompt_token_ids"] == [1, 2]
    engine.cfg.max_model_len = 1025
    with pytest.raises(ValueError, match="No se truncará"):
        engine.render([])


@pytest.mark.parametrize("show_progress", [False, True])
def test_bench_invokes_cli_and_preserves_raw_files_without_loading_models(
    tmp_path, monkeypatch, show_progress
):
    from common.files import read_json

    engine = LocalBench.__new__(LocalBench)
    engine.cfg, engine.run, engine.port = cfg(tmp_path), tmp_path, 8000
    engine.use_adapter, engine.counter = False, 0
    engine.render = lambda messages: {
        "prompt": "<chat>full prompt</chat>",
        "prompt_token_ids": [1, 2],
        "input_tokens": 2,
    }
    calls, waits, progress = [], [], []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            output = (
                Path(command[command.index("--result-dir") + 1])
                / command[command.index("--result-filename") + 1]
            )
            output.write_text(json.dumps(native()))

        def wait(self, timeout=None):
            waits.append(timeout)
            if timeout is not None and len(waits) == 1:
                raise subprocess.TimeoutExpired("vllm bench serve", timeout)
            return 0

    monkeypatch.setattr("common.inference.subprocess.Popen", Process)
    monkeypatch.setattr("common.inference.stop_process", lambda process: None)
    result = engine.batch(
        [[{"role": "user", "content": "question"}]],
        report=progress.append if show_progress else None,
    )
    if show_progress:
        assert waits == [30, 30]
        assert len(progress) == 1 and progress[0].startswith("vLLM sigue generando")
    else:
        assert waits == [None] and not progress
    assert result[0]["raw_response"] == native()["generated_texts"][0]
    command, options = calls[0]
    assert command[:3] == ["vllm", "bench", "serve"]
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    assert options["env"]["NO_PROXY"] == "127.0.0.1,localhost"
    assert "OPENAI_API_KEY" not in options["env"]
    assert read_json(tmp_path / "bench-0001/result.json") == native()
    assert (tmp_path / "bench-0001/prompts.jsonl").read_text().find("full prompt") > 0
    assert (tmp_path / "bench-0001/memory.json").exists()


def test_failed_bench_never_returns_fabricated_metrics(tmp_path, monkeypatch):
    engine = LocalBench.__new__(LocalBench)
    engine.cfg, engine.run, engine.port = cfg(tmp_path), tmp_path, 8000
    engine.use_adapter, engine.counter = False, 0
    engine.render = lambda messages: {"prompt": "x", "prompt_token_ids": [1], "input_tokens": 1}
    cleaned = []
    monkeypatch.setattr(
        "common.inference.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(wait=lambda: 2)
    )
    monkeypatch.setattr("common.inference.stop_process", lambda process: cleaned.append(process))
    with pytest.raises(RuntimeError, match="código 2"):
        engine.batch([[]])
    assert len(cleaned) == 1
    assert (tmp_path / "bench-0001/bench.log").exists()
    assert not (tmp_path / "bench-0001/result.json").exists()


def test_server_failure_keeps_compiler_root_cause_outside_last_100_lines(tmp_path):
    from common.inference import server_failure

    log = tmp_path / "serve.log"
    log.write_text(
        "/tmp/cuda_utils.c:4: fatal error: Python.h: No such file or directory\n"
        + "traceback frame\n" * 200
        + "RuntimeError: Engine core initialization failed\n"
    )
    message = server_failure(log, 1)
    assert "fatal error: Python.h" in message
    assert str(log) in message
    assert "código 1" in message


def test_busy_port_fails_with_actionable_message_without_stopping_any_process(monkeypatch):
    import errno

    from common.inference import check_local_port

    class BusySocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 8000)
            raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr("common.inference.socket.socket", BusySocket)
    with pytest.raises(RuntimeError, match="--port 8001"):
        check_local_port(8000)
