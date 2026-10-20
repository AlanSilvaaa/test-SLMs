"""Fixed OA-stratified split and stable demonstration assignments."""

import random
import re
from collections import defaultdict
from pathlib import Path

from common.contracts import Example, Request
from common.files import append_jsonl, load_rows, new_run, write_json


def normalized(value):
    return re.sub(r"\W+", " ", value.casefold()).strip()


def validate_separation(train, validation, benchmark):
    all_examples = train + validation
    keys = [(r["id"], r["version"]) for r in all_examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Hay solapamiento entre entrenamiento y validación")
    statements = [normalized(r["enunciado"]) for r in all_examples]
    if len(statements) != len(set(statements)):
        raise ValueError("Hay enunciados duplicados en el dataset")
    reserved = {normalized(r["solicitud"]) for r in benchmark}
    if any(normalized(r["solicitud"]) in reserved for r in all_examples):
        raise ValueError("Una solicitud del benchmark aparece en el dataset")
    if {r["id"] for r in benchmark} & {r["id"] for r in all_examples}:
        raise ValueError("Hay identificadores compartidos con el benchmark")


def prepare(dataset: Path, benchmark: Path, output: Path):
    examples = load_rows(dataset, Example)
    requests = load_rows(benchmark, Request)
    validate_separation(examples, [], requests)
    groups = defaultdict(list)
    for row in examples:
        groups[row["codigo_oa"]].append(row)
    if len(groups) != 59 or len(requests) != 295:
        raise ValueError("El protocolo requiere 59 OA y 295 solicitudes")
    counts = defaultdict(int)
    for row in requests:
        counts[row["codigo_oa"]] += 1
    if set(groups) != set(counts) or any(n != 5 for n in counts.values()):
        raise ValueError("Se requieren cinco solicitudes por OA del dataset")
    train, validation, demos = [], [], {}
    rng = random.Random(42)
    for oa, rows in sorted(groups.items()):
        if len(rows) != 40:
            raise ValueError(f"{oa}: se requieren 40 ejemplos aceptados")
        rows.sort(key=lambda r: (r["id"], r["version"]))
        rng.shuffle(rows)
        selected = rows[:34]
        train.extend(selected)
        validation.extend(rows[34:])
        # Prefer distinct skills, retaining deterministic ordering.
        diverse, rest, seen = [], [], set()
        for row in selected:
            skill = normalized(row["habilidad_evaluada"])
            if skill in seen:
                rest.append(row)
            else:
                diverse.append(row)
                seen.add(skill)
        demos[oa] = [{"id": r["id"], "version": r["version"]} for r in (diverse + rest)[:4]]
    run = new_run(output, "split", {"seed": 42}, [dataset, benchmark])
    for name, rows in (("train", train), ("validation", validation)):
        for row in rows:
            append_jsonl(run / f"{name}.jsonl", row)
    write_json(run / "demonstrations.json", demos)
    return run
