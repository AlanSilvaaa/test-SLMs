"""Official vLLM CLI processes, restricted to this experimental computer."""

import errno
import json
import math
import os
import signal
import socket
import subprocess
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

from common.files import append_jsonl, read_json, write_json
from common.monitor import MemoryMonitor
from common.runtime import check_inference_runtime

VLLM_VERSION = "0.15.1"
HOST = "127.0.0.1"
PROGRESS_INTERVAL_S = 30


def check_local_port(port):
    try:
        with socket.socket() as sock:
            # Match the server's reuse policy: TIME_WAIT is not a listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, port))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"El puerto {HOST}:{port} está ocupado. "
                "Usa --port 8001 en la prueba rápida o cambia port en la configuración. "
                f"Para identificar el proceso: ss -ltnp 'sport = :{port}'"
            ) from error
        raise


def offline_environment():
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(name, None)
    # The official benchmark reads this variable even for a local server.
    env.pop("OPENAI_API_KEY", None)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    return env


def serve_command(cfg, port, use_adapter=False):
    command = [
        "vllm",
        "serve",
        cfg.model.path,
        "--host",
        HOST,
        "--port",
        str(port),
        "--revision",
        cfg.model.revision,
        "--tokenizer-revision",
        cfg.model.revision,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--max-num-seqs",
        "1",
        "--seed",
        str(cfg.seed),
        "--generation-config",
        "vllm",
        "--no-enable-prefix-caching",
    ]
    if use_adapter:
        command += [
            "--enable-lora",
            "--max-lora-rank",
            "16",
            "--lora-modules",
            f"math={cfg.adapter}",
        ]
    return command


def bench_command(
    cfg, port, dataset, result, count, seed, temperature, schema=None, use_adapter=False
):
    body = {"seed": seed, "add_special_tokens": False}
    if schema is not None:
        body["structured_outputs"] = {"json": schema}
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        HOST,
        "--port",
        str(port),
        "--backend",
        "vllm",
        "--endpoint",
        "/v1/completions",
        "--model",
        cfg.model.path,
        "--tokenizer",
        cfg.model.path,
        "--tokenizer-mode",
        "hf",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset),
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
        "--num-prompts",
        str(count),
        "--custom-output-len",
        str(cfg.max_tokens),
        "--max-concurrency",
        "1",
        "--request-rate",
        "inf",
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--top-p",
        str(cfg.top_p),
        "--extra-body",
        json.dumps(body),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result.parent),
        "--result-filename",
        result.name,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--ready-check-timeout-sec",
        str(cfg.ready_timeout_s),
        "--num-warmups",
        "0",
    ]
    if use_adapter:
        # The single registered module is selected for EVERY request, including
        # the benchmark's initial readiness request. Do not use random modules.
        command += ["--served-model-name", "math"]
    return command


def read_detailed(result, input_counts, max_tokens):
    """v0.15.1 detailed arrays are in input order, including failed requests."""
    fields = ("generated_texts", "errors", "ttfts", "itls", "output_lens", "input_lens")
    count = len(input_counts)
    if any(not isinstance(result.get(k), list) or len(result[k]) != count for k in fields):
        raise ValueError("Resultado bench incompleto o incompatible: faltan arrays por solicitud")
    rows = []
    for index, input_tokens in enumerate(input_counts):
        error = result["errors"][index] or None
        output = result["output_lens"][index]
        ttft = result["ttfts"][index]
        itls = result["itls"][index]
        if (
            type(output) is not int
            or output < 0
            or not isinstance(itls, list)
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in [ttft, *itls]
            )
        ):
            raise ValueError("Métricas bench inválidas")
        decode = sum(itls) if not error else None
        latency = ttft + decode if decode is not None else None
        limited = output >= max_tokens
        rows.append(
            {
                "raw_response": result["generated_texts"][index],
                "error": error,
                "output_limit_reached": limited,
                # v0.15.1 does not retain finish_reason in its detailed JSON.
                "finish_reason": None,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output,
                    "total_tokens": input_tokens + output,
                },
                "bench_input_tokens": result["input_lens"][index],
                "latency_s": latency,
                "ttft_s": ttft if not error else None,
                "decode_s": decode,
                "tpot_s": decode / (output - 1) if decode and output > 1 else None,
                "decode_tokens_per_second": (output - 1) / decode
                if decode and output > 1
                else None,
                "timing_source": "vllm_bench_serve_stream",
            }
        )
    return rows


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def server_failure(log_path, exit_code):
    """Retain early compiler diagnostics that disappear from a traceback tail."""
    diagnostics, tail = [], deque(maxlen=8)
    needles = (
        "fatal error:",
        "cannot find -l",
        "undefined reference",
        "collect2:",
        "importerror:",
        "modulenotfounderror:",
        "error while loading shared libraries",
        "permission denied",
        "runtimeerror:",
    )
    try:
        with Path(log_path).open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tail.append(line[:1200])
                if len(diagnostics) < 8 and any(n in line.lower() for n in needles):
                    diagnostics.append(line[:1200])
    except OSError:
        pass
    excerpt = "\n".join(diagnostics or tail)
    return f"vllm serve terminó con código {exit_code}. Log: {log_path}" + (
        f"\n{excerpt}" if excerpt else ""
    )


class LocalBench:
    """No model inference API or custom HTTP client: invoke vllm commands."""

    def __init__(self, cfg, run, *, use_adapter=False):
        check_inference_runtime()
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Prepara el equipo experimental con uv sync --extra infer"
            ) from error

        installed = version("vllm")
        if installed.split("+", 1)[0] != VLLM_VERSION or "+cpu" in installed:
            raise ValueError(f"Este protocolo requiere vLLM {VLLM_VERSION}; revisa el lockfile")
        if not Path(cfg.model.path).is_dir():
            raise ValueError("Se requieren pesos y tokenizador locales")
        self.cfg, self.run, self.use_adapter = cfg, Path(run), use_adapter
        self.installed_version = installed
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.path, revision=cfg.model.revision, local_files_only=True
        )
        self.process = None
        self.log = None
        self.counter = 0
        self.port = cfg.port
        write_json(
            self.run / "tokenizer.json",
            {
                "chat_template": self.tokenizer.chat_template,
                "chat_template_kwargs": cfg.chat_template_kwargs,
                "special_tokens_map": self.tokenizer.special_tokens_map,
            },
        )

    def render(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.cfg.chat_template_kwargs
        )
        if ids != expected:
            raise ValueError("La plantilla textual no reproduce los tokens previstos")
        if len(ids) + self.cfg.max_tokens > self.cfg.max_model_len:
            raise ValueError("Entrada + salida excede el contexto. No se truncará")
        return {"prompt": prompt, "prompt_token_ids": ids, "input_tokens": len(ids)}

    def tokenize(self, messages):
        return self.render(messages)["input_tokens"]

    def preflight(self, messages):
        return self.tokenize(messages)

    def metadata(self):
        return {
            "execution": "local_cli",
            "vllm_version": self.installed_version,
            "host": HOST,
            "port": self.port,
            "dtype": "bfloat16",
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "max_concurrency": 1,
            "enable_prefix_caching": False,
            "initial_readiness_request": True,
            "num_warmups": 0,
        }

    def start(self, *, report=None):
        # Fail on an occupied port rather than benchmarking an unrelated service.
        check_local_port(self.port)
        command = serve_command(self.cfg, self.port, self.use_adapter)
        append_jsonl(self.run / "commands.jsonl", {"role": "serve", "argv": command})
        self.log = (self.run / "serve.log").open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=offline_environment(),
            start_new_session=True,
        )
        started = time.monotonic()
        deadline = started + self.cfg.ready_timeout_s
        next_report = started + PROGRESS_INTERVAL_S
        while (now := time.monotonic()) < deadline:
            if report and now >= next_report:
                report(f"vLLM sigue iniciando ({int(now - started)} s transcurridos).")
                next_report = now + PROGRESS_INTERVAL_S
            if self.process.poll() is not None:
                raise RuntimeError(server_failure(self.run / "serve.log", self.process.returncode))
            try:
                with socket.create_connection((HOST, self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("vllm serve no abrió el puerto dentro del plazo; revisa serve.log")

    def batch(self, messages, *, seed=None, temperature=None, schema=None, report=None):
        rendered = [self.render(m) for m in messages]
        self.counter += 1
        folder = self.run / f"bench-{self.counter:04d}"
        folder.mkdir(exist_ok=False)
        dataset, output = folder / "prompts.jsonl", folder / "result.json"
        for item in rendered:
            append_jsonl(dataset, {"prompt": item["prompt"]})
        write_json(folder / "inputs.json", rendered)
        command = bench_command(
            self.cfg,
            self.port,
            dataset,
            output,
            len(rendered),
            self.cfg.seed if seed is None else seed,
            self.cfg.temperature if temperature is None else temperature,
            schema,
            self.use_adapter,
        )
        append_jsonl(self.run / "commands.jsonl", {"role": "bench", "argv": command})
        process = None
        with (folder / "bench.log").open("x", encoding="utf-8") as log:
            try:
                with MemoryMonitor() as memory:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=offline_environment(),
                        start_new_session=True,
                    )
                    if report:
                        started = time.monotonic()
                        while True:
                            try:
                                code = process.wait(timeout=PROGRESS_INTERVAL_S)
                                break
                            except subprocess.TimeoutExpired:
                                elapsed = int(time.monotonic() - started)
                                report(f"vLLM sigue generando ({elapsed} s transcurridos).")
                    else:
                        code = process.wait()
                write_json(folder / "memory.json", memory.result())
            finally:
                stop_process(process)
        if code:
            raise RuntimeError(f"vllm bench serve terminó con código {code}; revisa {folder}")
        raw = read_json(output)
        rows = read_detailed(raw, [r["input_tokens"] for r in rendered], self.cfg.max_tokens)
        for index, row in enumerate(rows):
            row["bench_result"] = str(output.resolve())
            row["bench_index"] = index
        return rows

    def complete(self, messages, schema=None, *, seed=None, temperature=None):
        return self.batch([messages], seed=seed, temperature=temperature, schema=schema)[0]

    def close(self):
        stop_process(self.process)
        if self.log:
            self.log.close()
