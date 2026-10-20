"""Exercise the Python API with fake modules, without importing real vLLM."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from common.contracts import Experiment
from common.inference import LocalVLLM, request_timings


@pytest.fixture()
def fake_vllm(monkeypatch, tmp_path):
    calls = {"loads": [], "generate": [], "templates": []}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["templates"].append((messages, kwargs))
            return [1, 2, 3]

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["loads"].append(kwargs)

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, prompts, **kwargs):
            calls["generate"].append((prompts, kwargs))
            return [
                SimpleNamespace(
                    finished=True,
                    metrics=None,
                    outputs=[
                        SimpleNamespace(
                            text='{"enunciado":"P","solucion":"S"}',
                            token_ids=[10, 11],
                            finish_reason="stop",
                        )
                    ],
                )
            ]

    def options(**kwargs):
        return SimpleNamespace(**kwargs)

    module = ModuleType("vllm")
    module.LLM, module.SamplingParams, module.__version__ = FakeLLM, options, "fake"
    sampling = ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = options
    lora = ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda name, identifier, path: (name, identifier, path)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)
    cfg = Experiment(
        model={"path": str(tmp_path), "revision": "abc"},
        benchmark="unused",
        max_tokens=10,
        max_model_len=20,
        chat_template_kwargs={"enable_thinking": False},
    )
    return cfg, calls


def test_local_loading_and_generation(fake_vllm):
    cfg, calls = fake_vllm
    engine = LocalVLLM(cfg)
    try:
        assert engine.preflight([]) == 3
        for _ in range(2):
            result = engine.complete([], {"type": "object"}, seed=7, temperature=0)
            assert result["finish_reason"] == "stop"
            assert result["usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
            assert result["latency_s"] >= 0
            assert result["ttft_s"] is None
            assert result["decode_tokens_per_second"] is None
        assert len(calls["loads"]) == 1
        assert calls["loads"][0]["model"] == cfg.model.path
        assert calls["loads"][0]["enable_lora"] is False
        prompts, options = calls["generate"][0]
        assert prompts == [{"prompt_token_ids": [1, 2, 3]}]
        assert options["sampling_params"].structured_outputs.json == {"type": "object"}
        assert options["sampling_params"].seed == 7
        assert options["sampling_params"].temperature == 0
        assert calls["templates"][0][1]["enable_thinking"] is False
        assert engine.metadata()["execution"] == "local"
    finally:
        engine.close()


def test_local_context_overflow_never_generates(fake_vllm):
    cfg, calls = fake_vllm
    cfg.max_model_len = 12
    engine = LocalVLLM(cfg)
    try:
        with pytest.raises(ValueError, match="excede"):
            engine.preflight([])
        with pytest.raises(ValueError, match="No se truncará"):
            engine.complete([], {})
        assert calls["generate"] == []
    finally:
        engine.close()


def test_adapter_loaded_directly_for_lora(fake_vllm):
    cfg, calls = fake_vllm
    cfg.adapter = "/local/adapter"
    engine = LocalVLLM(cfg, use_adapter=True)
    try:
        engine.complete([], {})
        assert calls["loads"][0]["enable_lora"] is True
        assert calls["loads"][0]["max_lora_rank"] == 16
        assert calls["generate"][0][1]["lora_request"] == ("math", 1, "/local/adapter")
    finally:
        engine.close()


def test_engine_metrics_are_optional_and_checked():
    assert request_timings(None, 2)["ttft_s"] is None
    times = SimpleNamespace(arrival_time=1.0, first_token_time=1.5, last_token_time=2.0)
    assert request_timings(times, 3) == {
        "ttft_s": 0.5,
        "decode_s": 0.5,
        "decode_tokens_per_second": 4.0,
        "timing_source": "vllm_request_metrics",
    }
    times.last_token_time = 1.0
    assert request_timings(times, 3)["decode_s"] is None


def test_configuration_has_no_connection_fields():
    assert "base_url" not in Experiment.model_fields
    assert "served_name" not in Experiment.model_fields["model"].annotation.model_fields
