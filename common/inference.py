"""Run vLLM directly in the current computer, without an HTTP service."""

import gc
from pathlib import Path
from time import perf_counter


class LocalVLLM:
    """Share local model loading, tokenization and generation across experiments."""

    def __init__(self, cfg, *, use_adapter=False):
        # Lightweight commands and unit tests do not require CUDA or vLLM.
        try:
            from vllm import LLM, SamplingParams, __version__
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as error:
            raise RuntimeError(
                "Instala vLLM en el equipo experimental con uv sync --extra infer"
            ) from error
        if not Path(cfg.model.path).is_dir():
            raise ValueError(f"Se requieren pesos locales en {cfg.model.path}")
        if use_adapter and not cfg.adapter:
            raise ValueError("Falta la ruta local del adaptador LoRA")
        self.cfg = cfg
        self.version = __version__
        self.sampling_params = SamplingParams
        self.structured_outputs = StructuredOutputsParams
        self.lora_request = None
        self.llm = LLM(
            model=cfg.model.path,
            revision=cfg.model.revision,
            tokenizer_revision=cfg.model.revision,
            dtype="bfloat16",
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=1,
            max_num_seqs=1,
            seed=cfg.seed,
            generation_config="vllm",
            enable_prefix_caching=False,
            enable_lora=use_adapter,
            **({"max_lora_rank": 16} if use_adapter else {}),
        )
        self.tokenizer = self.llm.get_tokenizer()
        if use_adapter:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("math", 1, cfg.adapter)

    def close(self):
        # Each CLI command owns one engine. Its workers exit with this process.
        self.tokenizer = None
        self.llm = None
        gc.collect()

    def metadata(self):
        return {
            "execution": "local",
            "vllm_version": self.version,
            "model": self.cfg.model.model_dump(),
            "adapter": self.cfg.adapter if self.lora_request else None,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
        }

    def token_ids(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **self.cfg.chat_template_kwargs,
        )

    def tokenize(self, messages):
        return len(self.token_ids(messages))

    def preflight(self, messages):
        count = self.tokenize(messages)
        if count + self.cfg.max_tokens > self.cfg.max_model_len:
            raise ValueError(
                f"Entrada {count} + salida {self.cfg.max_tokens} excede {self.cfg.max_model_len}"
            )
        return count

    def complete(self, messages, schema, *, seed=None, temperature=None):
        token_ids = self.token_ids(messages)
        if len(token_ids) + self.cfg.max_tokens > self.cfg.max_model_len:
            raise ValueError("La solicitud excede el contexto. No se truncará")
        params = self.sampling_params(
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            seed=self.cfg.seed if seed is None else seed,
            structured_outputs=self.structured_outputs(json=schema),
        )
        started = perf_counter()
        outputs = self.llm.generate(
            [{"prompt_token_ids": token_ids}],
            sampling_params=params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        elapsed = perf_counter() - started
        if len(outputs) != 1 or len(outputs[0].outputs) != 1 or not outputs[0].finished:
            raise ValueError("vLLM devolvió una salida incompleta o una cantidad inesperada")
        request = outputs[0]
        completion = request.outputs[0]
        output_tokens = len(completion.token_ids)
        return {
            "raw_response": completion.text,
            "finish_reason": completion.finish_reason,
            "usage": {
                "prompt_tokens": len(token_ids),
                "completion_tokens": output_tokens,
                "total_tokens": len(token_ids) + output_tokens,
            },
            "latency_s": elapsed,
            "output_tokens_per_wall_second": output_tokens / elapsed if elapsed > 0 else None,
            **request_timings(getattr(request, "metrics", None), output_tokens),
        }


def request_timings(metrics, output_tokens):
    """Only report per-token timings when the engine actually supplies them."""
    first = getattr(metrics, "first_token_time", None)
    last = getattr(metrics, "last_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    ttft = (
        first - arrival if first is not None and arrival is not None and first >= arrival else None
    )
    decode = last - first if last is not None and first is not None and last > first else None
    return {
        "ttft_s": ttft,
        "decode_s": decode,
        "decode_tokens_per_second": (output_tokens - 1) / decode
        if decode and output_tokens > 1
        else None,
        "timing_source": "vllm_request_metrics" if ttft is not None else "unavailable",
    }
