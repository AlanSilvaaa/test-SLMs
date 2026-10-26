from pathlib import Path
from types import SimpleNamespace

import pytest

from common.runtime import validate_torch_runtime


def test_cpu_torch_is_rejected_with_actionable_error():
    torch = SimpleNamespace(__version__="2.9.1+cpu", version=SimpleNamespace(cuda=None))
    with pytest.raises(RuntimeError, match="solo CPU"):
        validate_torch_runtime(torch)


def test_cuda_build_without_device_is_distinguished_from_cpu_build():
    torch = SimpleNamespace(
        __version__="2.9.1+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(RuntimeError, match="no puede acceder"):
        validate_torch_runtime(torch)
    torch.cuda.is_available = lambda: True
    validate_torch_runtime(torch)


def test_cpu_runtime_failure_precedes_model_download(monkeypatch):
    from common.quick import run_quick

    def cpu_error():
        raise RuntimeError("solo CPU")

    monkeypatch.setattr("common.quick.check_inference_runtime", cpu_error)
    monkeypatch.setattr("common.quick.prepare_model", lambda *a, **k: pytest.fail("download"))
    with pytest.raises(RuntimeError, match="solo CPU"):
        run_quick(5)


def test_local_bench_accepts_official_cuda_build_and_records_full_version(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from common.contracts import Experiment
    from common.inference import LocalBench

    fake = ModuleType("transformers")
    fake.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **k: SimpleNamespace(
            chat_template="template", special_tokens_map={}
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr("common.inference.check_inference_runtime", lambda: None)
    monkeypatch.setattr("common.inference.version", lambda name: "0.15.1+cu130")
    cfg = Experiment(model={"path": str(tmp_path), "revision": "a" * 40}, benchmark="unused")
    bench = LocalBench(cfg, tmp_path)
    assert bench.metadata()["vllm_version"] == "0.15.1+cu130"
    monkeypatch.setattr("common.inference.version", lambda name: "0.15.1+cpu")
    with pytest.raises(ValueError, match="requiere vLLM"):
        LocalBench(cfg, tmp_path)


def test_lock_pins_arm64_cuda_and_retains_x86_package_source():
    import tomllib

    root = Path(__file__).resolve().parent.parent
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = lock["package"]
    torch = [p for p in packages if p["name"] == "torch" and p["version"] == "2.9.1+cu130"]
    assert len(torch) == 1
    assert torch[0]["source"]["registry"] == "https://download.pytorch.org/whl/cu130"
    vllm = [p for p in packages if p["name"] == "vllm" and p["version"] == "0.15.1+cu130"]
    assert len(vllm) == 1
    assert "aarch64.whl" in vllm[0]["source"]["url"]
    assert any(
        p["name"] == "vllm"
        and p["version"] == "0.15.1"
        and p["source"].get("registry") == "https://pypi.org/simple"
        for p in packages
    )
