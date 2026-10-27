"""Sequential overnight pilot on the experimental host only."""

import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from common.contracts import Experiment, Model
from common.files import new_run, read_json, write_json
from common.models import DEFAULT_MODELS_DIR, prepare_model
from common.pilot_data import DEFAULT_PILOT, ROOT, validate_pilot


def launch_pilot(data, output, models_dir, offline=False, port=8000):
    """Detach the local CLI with closed stdin and an exclusive, persistent log."""
    data, output, models_dir = (Path(p).resolve() for p in (data, output, models_dir))
    # Fail visibly for an invalid/missing package before starting a detached job.
    cfg = Experiment(
        model=Model(path="not-loaded", revision="not-loaded"),
        phase="pilot",
        benchmark=str(data / "benchmark.jsonl"),
        oa_registry=str(data / "oa.json"),
        pilot_manifest=str(data / "manifest.json"),
        development=str(ROOT / "examples/smoke/requests.jsonl"),
        repetitions=1,
    )
    validate_pilot(cfg)
    logs = output / "tandas-piloto"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"lanzamiento-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "pilot",
        "--data",
        str(data),
        "--output",
        str(output),
        "--models-dir",
        str(models_dir),
        "--port",
        str(port),
    ]
    if offline:
        command.append("--offline")
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return process.pid, log


def run_pilot(
    data=DEFAULT_PILOT,
    output=Path("output"),
    models_dir=DEFAULT_MODELS_DIR,
    offline=False,
    port=8000,
    report=print,
):
    from common.generation import build_requests, infer
    from common.inference import check_local_port
    from common.runtime import check_inference_runtime

    data, output = Path(data).resolve(), Path(output).resolve()
    catalog = read_json(ROOT / "configs/models.json")["generators"]
    cfg = Experiment(
        model=Model(id=catalog[0], path="not-loaded", revision="not-loaded"),
        phase="pilot",
        benchmark=str(data / "benchmark.jsonl"),
        oa_registry=str(data / "oa.json"),
        pilot_manifest=str(data / "manifest.json"),
        development=str(ROOT / "examples/smoke/requests.jsonl"),
        repetitions=1,
        output=str(output),
        port=port,
    )
    # Validate the full package before runtime discovery or any model download.
    inputs = validate_pilot(cfg)
    build_requests(cfg, "base_model", 0)
    check_inference_runtime()
    check_local_port(port)
    batch = new_run(
        output,
        "tandas-piloto",
        {
            "phase": "pilot",
            "models": catalog,
            "requests_per_model": 295,
            "repetitions": 1,
            "offline": offline,
            "port": port,
        },
        inputs,
    )
    state = {"status": "running", "phase": "pilot", "models": []}
    write_json(batch / "status.json", state)
    report(f"Tanda piloto: {batch}")
    report("6 modelos base × 295 solicitudes × 1 pasada = 1770 generaciones; sin judge.")
    try:
        for index, identifier in enumerate(catalog, 1):
            record = {"model": identifier, "status": "running"}
            state["models"].append(record)
            write_json(batch / "status.json", state)
            report(
                f"[{index}/{len(catalog)}] {identifier}: preparando pesos e iniciando generación"
            )
            try:
                check_local_port(port)
                cfg.model = prepare_model(identifier, models_dir, offline=offline)
                cfg.chat_template_kwargs = (
                    {"enable_thinking": False} if "SmolLM3" in identifier else {}
                )
                run = infer(cfg, "base_model", 0)
                record.update(status="complete", run=str(run), excel=str(run / "resultados.xlsx"))
                report(f"Excel: {run / 'resultados.xlsx'}")
            except Exception as error:
                error_log = batch / f"error-{index:02d}.log"
                error_log.write_text(traceback.format_exc(), encoding="utf-8")
                record.update(status="failed", error=str(error), log=str(error_log))
                report(
                    f"Error en {identifier}: {error}. Continúa con el siguiente; registro: {error_log}"
                )
            write_json(batch / "status.json", state)
    except BaseException:
        state["status"] = "interrupted"
        if state["models"] and state["models"][-1]["status"] == "running":
            state["models"][-1]["status"] = "interrupted"
        write_json(batch / "status.json", state)
        raise
    failed = sum(row["status"] == "failed" for row in state["models"])
    state["status"] = "completed_with_errors" if failed else "complete"
    write_json(batch / "status.json", state)
    report(
        f"Tanda finalizada: {len(catalog) - failed} modelos completados; {failed} fallidos. {batch}"
    )
    if failed:
        raise RuntimeError(
            f"La tanda terminó con {failed} modelos fallidos; revisa {batch / 'status.json'}"
        )
    return batch
