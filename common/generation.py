"""Shared runner. Each strategy only owns construction of its messages."""

from collections import defaultdict
from importlib import import_module
from pathlib import Path

from pydantic import ValidationError

from common.contracts import Example, GeneratedProblem, Request
from common.data import validate_development, validate_requests, validate_separation
from common.files import append_jsonl, load_rows, new_run, read_json, sha256, write_json
from common.inference import LocalBench
from common.monitor import MemoryMonitor
from common.prompts import conversation


def build_requests(cfg, module, shots):
    development = load_rows(cfg.development, Request)
    benchmark_path = Path(cfg.benchmark)
    # A smoke test may precede construction of the final reserved benchmark.
    # If that register exists, always check separation; final runs require it.
    if cfg.phase == "benchmark" or benchmark_path.exists():
        benchmark = load_rows(benchmark_path, Request)
    else:
        benchmark = []
    validate_development(development, benchmark)
    inputs = [Path(cfg.development)]
    if benchmark:
        inputs.append(benchmark_path)
    if cfg.phase == "benchmark":
        if cfg.request_limit is not None:
            raise ValueError("No se permite limitar solicitudes del benchmark final")
        validate_requests(benchmark, cfg.oa_registry)
        inputs.append(Path(cfg.oa_registry))
        if cfg.repetitions < 2:
            raise ValueError("Las mediciones finales requieren al menos dos repeticiones")
        if not cfg.load_conditions.strip():
            raise ValueError("Registra las condiciones de carga del computador experimental")
    requests = benchmark if cfg.phase == "benchmark" else development
    if cfg.request_limit is not None:
        if cfg.request_limit > len(requests):
            raise ValueError("El límite solicitado supera las solicitudes disponibles")
        requests = requests[: cfg.request_limit]
    context = {}
    if module == "few-shot":
        train = load_rows(cfg.train, Example)
        validation = load_rows(cfg.validation, Example)
        validate_separation(train, validation, benchmark)
        validate_separation(train, validation, development)
        context = {
            "examples": {(r["id"], r["version"]): r for r in train},
            "assignments": read_json(cfg.demonstrations),
            "shots": shots,
        }
        inputs.extend(map(Path, [cfg.train, cfg.validation, cfg.demonstrations]))
    if module == "skill":
        context["instructions"] = Path(cfg.skill).read_text(encoding="utf-8")
        inputs.append(Path(cfg.skill))
        if cfg.phase == "benchmark":
            review = read_json(cfg.skill_review)
            if (
                review.get("status") != "approved"
                or review.get("skill_sha256") != sha256(Path(cfg.skill))
                or not review.get("pedagogical_review")
                or not review.get("development_runs")
            ):
                raise ValueError("Falta revisión pedagógica y pruebas de desarrollo de la skill")
            inputs.append(Path(cfg.skill_review))
    if module == "lora":
        if not cfg.adapter or not cfg.adapter_sha256:
            raise ValueError("LoRA requiere adapter y adapter_sha256 en la configuración")
        adapter = Path(cfg.adapter)
        if sha256(adapter / "adapter_model.safetensors") != cfg.adapter_sha256:
            raise ValueError("El checksum del adaptador no coincide")
        provenance = read_json(adapter / "provenance.json")
        if (
            provenance["base_revision"] != cfg.model.revision
            or provenance.get("base_id") != cfg.model.id
            or not cfg.model.id
        ):
            raise ValueError("El adaptador corresponde a otra revisión del modelo base")
        inputs.extend(
            [
                adapter / "adapter_config.json",
                adapter / "adapter_model.safetensors",
                adapter / "provenance.json",
            ]
        )
    builder = import_module(module.replace("-", "_") + ".infer").messages
    ordinals = defaultdict(int)
    built = []
    for row in requests:
        ordinal = ordinals[row["codigo_oa"]]
        messages, metadata = builder(row, ordinal=ordinal, **context)
        ordinals[row["codigo_oa"]] += 1
        built.append(
            {
                "request": row,
                "messages": messages,
                "benchmark_separation_checked": bool(benchmark),
                **metadata,
            }
        )
    return built, inputs


def infer(cfg, module, shots):
    from common.performance import summarize_performance
    from common.runtime import validate_model

    validate_model(cfg, judge=False)
    shots = shots if module == "few-shot" else 0
    built, inputs = build_requests(cfg, module, shots)
    run = new_run(
        cfg.output, module, {**cfg.model_dump(), "module": module, "shots": shots}, inputs
    )
    engine = LocalBench(cfg, run, use_adapter=module == "lora")
    try:
        write_json(run / "engine.json", engine.metadata())
        for item in built:
            item.update(engine.render(item["messages"]))
            base_count = engine.tokenize(conversation(item["request"]))
            item["additional_input_tokens"] = item["input_tokens"] - base_count
            append_jsonl(run / "requests.jsonl", item)
        # Includes the pre-load baseline and startup peak, separate from each bench.
        with MemoryMonitor() as memory:
            engine.start()
            for repetition in range(cfg.repetitions):
                results = engine.batch(
                    [item["messages"] for item in built], seed=cfg.seed + repetition
                )
                for item, result in zip(built, results, strict=True):
                    row = {
                        **item,
                        **result,
                        "module": module,
                        "shots": shots,
                        "repetition": repetition,
                        "model": cfg.model.model_dump(),
                        "phase": cfg.phase,
                        "problem": None,
                        "parse_error": None,
                    }
                    if not row["error"]:
                        try:
                            row["problem"] = GeneratedProblem.model_validate_json(
                                row["raw_response"]
                            ).model_dump()
                        except (ValueError, ValidationError) as error:
                            row["parse_error"] = str(error)
                    append_jsonl(run / "generations.jsonl", row)
        write_json(run / "memory.json", memory.result())
    finally:
        engine.close()
    summarize_performance(run / "generations.jsonl", run / "performance.json")
    write_json(run / "status.json", {"status": "complete"})
    return run
