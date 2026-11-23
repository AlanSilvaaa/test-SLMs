"""Small base-model development runs without editing configuration files."""

from pathlib import Path

from common.contracts import Experiment, Model, Request
from common.files import load_rows
from common.inference import check_local_port
from common.models import DEFAULT_MODELS_DIR, prepare_model
from common.runtime import check_inference_runtime

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUESTS = ROOT / "examples/smoke/requests.jsonl"


def run_quick(
    count=5,
    model="SmolLM3",
    models_dir=DEFAULT_MODELS_DIR,
    revision=None,
    offline=False,
    requests=DEFAULT_REQUESTS,
    output=Path("output"),
    port=8000,
):
    # Validate input count before any possible download or GPU operation.
    rows = load_rows(requests, Request)
    if count > len(rows):
        raise ValueError(
            f"Pediste {count} problemas, pero {requests} contiene {len(rows)} solicitudes. "
            "Usa --requests con un archivo que tenga suficientes solicitudes"
        )
    check_inference_runtime()
    check_local_port(port)
    prepared = prepare_model(model, models_dir, revision, offline=offline)
    cfg = Experiment(
        model=prepared,
        phase="development",
        benchmark=str(ROOT / "data/benchmark.jsonl"),
        development=str(Path(requests).resolve()),
        request_limit=count,
        output=str(Path(output).resolve()),
        repetitions=1,
        port=port,
        chat_template_kwargs={"enable_thinking": False} if "SmolLM3" in prepared.id else {},
    )
    from common.generation import infer

    return infer(cfg, "base_model", 0)


def run_benchmark_base(
    benchmark, model, models_dir, revision, offline, output, port, check_only, *, report=None
):
    from common.generation import build_requests, infer

    cfg = Experiment(
        model=Model(path="not-loaded", revision="not-loaded"),
        phase="benchmark_candidate",
        benchmark=str(Path(benchmark).resolve()),
        development=str(DEFAULT_REQUESTS),
        output=str(Path(output).resolve()),
        repetitions=1,
        port=port,
    )
    if report:
        report(f"Validando solicitudes en {cfg.benchmark}")
    built, _ = build_requests(cfg, "base_model", 0)
    count, version = len(built), built[0]["request"]["version"]
    if check_only:
        return count, version
    if report:
        report(f"Benchmark {version} validado ({count} solicitudes). Preparando pesos de {model}.")
    check_inference_runtime()
    check_local_port(port)
    cfg.model = prepare_model(model, models_dir, revision, offline=offline)
    if "SmolLM3" in (cfg.model.id or ""):
        cfg.chat_template_kwargs = {"enable_thinking": False}
    return infer(cfg, "base_model", 0, report=report)
