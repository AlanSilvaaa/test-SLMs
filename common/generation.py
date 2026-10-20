"""Shared runner. Each strategy only owns construction of its messages."""

from collections import defaultdict
from importlib import import_module
from pathlib import Path

from pydantic import ValidationError

from common.contracts import Example, Problem, Request
from common.data import validate_separation
from common.files import append_jsonl, load_rows, new_run, read_json, sha256, write_json
from common.inference import LocalVLLM
from common.monitor import MemoryMonitor
from common.prompts import conversation


def build_requests(cfg, module, shots):
    requests = load_rows(cfg.benchmark, Request)
    inputs = [Path(cfg.benchmark)]
    context = {}
    if module == "few-shot":
        train = load_rows(cfg.train, Example)
        validation = load_rows(cfg.validation, Example)
        validate_separation(train, validation, requests)
        context = {
            "examples": {(r["id"], r["version"]): r for r in train},
            "assignments": read_json(cfg.demonstrations),
            "shots": shots,
        }
        inputs.extend(map(Path, [cfg.train, cfg.validation, cfg.demonstrations]))
    if module == "skill":
        context["instructions"] = Path(cfg.skill).read_text(encoding="utf-8")
        inputs.append(Path(cfg.skill))
    if module == "lora":
        if not cfg.adapter or not cfg.adapter_sha256:
            raise ValueError("LoRA requiere adapter y adapter_sha256 en la configuración")
        adapter = Path(cfg.adapter)
        if sha256(adapter / "adapter_model.safetensors") != cfg.adapter_sha256:
            raise ValueError("El checksum del adaptador no coincide")
        provenance = read_json(adapter / "provenance.json")
        if provenance["base_revision"] != cfg.model.revision:
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
        built.append({"request": row, "messages": messages, **metadata})
    return built, inputs


def infer(cfg, module, shots):
    shots = shots if module == "few-shot" else 0
    built, inputs = build_requests(cfg, module, shots)
    run = new_run(
        cfg.output, module, {**cfg.model_dump(), "module": module, "shots": shots}, inputs
    )
    engine = LocalVLLM(cfg, use_adapter=module == "lora")
    try:
        write_json(run / "engine.json", engine.metadata())
        # Check every input before generating any response. Never silently truncate.
        for item in built:
            item["input_tokens"] = engine.preflight(item["messages"])
            base_count = engine.tokenize(conversation(item["request"]))
            item["additional_input_tokens"] = item["input_tokens"] - base_count
            append_jsonl(run / "requests.jsonl", item)
        for repetition in range(cfg.repetitions):
            for item in built:
                row = {
                    **item,
                    "module": module,
                    "shots": shots,
                    "repetition": repetition,
                    "model": cfg.model.model_dump(),
                }
                with MemoryMonitor() as memory:
                    try:
                        result = engine.complete(
                            item["messages"],
                            Problem.model_json_schema(),
                            seed=cfg.seed + repetition,
                        )
                        row.update(result)
                        if result["finish_reason"] != "stop":
                            raise ValueError(f"Salida incompleta: {result['finish_reason']}")
                        row["problem"] = Problem.model_validate_json(
                            result["raw_response"]
                        ).model_dump()
                        row["error"] = None
                    except (ValueError, ValidationError) as error:
                        row["error"] = str(error)
                row["memory"] = memory.result()
                append_jsonl(run / "generations.jsonl", row)
    finally:
        engine.close()
    write_json(run / "status.json", {"status": "complete"})
    return run
