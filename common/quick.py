"""Small base-model development runs without editing configuration files."""

from pathlib import Path

from common.contracts import Experiment, Request
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
    output=Path("output/smoke"),
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
