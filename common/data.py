"""Fixed OA-stratified split and stable demonstration assignments."""

import random
import re
from collections import defaultdict
from pathlib import Path

from common.contracts import Example, Request
from common.files import append_jsonl, load_rows, new_run, read_json, write_json


def normalized(value):
    return re.sub(r"\W+", " ", value.casefold()).strip()


def validate_separation(train, validation, benchmark):
    all_examples = train + validation
    keys = [r["id"] for r in all_examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Hay solapamiento de IDs o versiones entre entrenamiento y validación")
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


def validate_requests(requests, registry_path, *, require_approved=True):
    """Check the frozen OA register, not just an arbitrary count of codes."""
    registry = read_json(registry_path)
    if (require_approved and registry.get("status") != "approved") or not registry.get("version"):
        raise ValueError("El registro de OA debe estar revisado, aprobado y versionado")
    objectives = registry["objectives"]
    by_code = {r["codigo_oa"]: r for r in objectives}
    if len(objectives) != 59 or len(by_code) != 59:
        raise ValueError("El registro requiere los 59 OA seleccionados")
    # Course is stored canonically as the strings '1' through '6'.
    from collections import Counter

    expected = {
        "1": {"numeros": 3, "medicion": 1},
        "2": {"numeros": 4, "medicion": 1},
        "3": {"numeros": 6, "medicion": 2},
        "4": {"numeros": 9, "medicion": 3, "patrones": 1},
        "5": {"numeros": 12, "medicion": 2, "patrones": 1, "datos": 3},
        "6": {"numeros": 8, "medicion": 2, "patrones": 1},
    }
    actual = Counter((r["curso"], r["eje"]) for r in objectives)
    target = Counter(
        {(course, axis): n for course, axes in expected.items() for axis, n in axes.items()}
    )
    if actual != target:
        raise ValueError("La distribución curso/eje no coincide con la tabla de la memoria")
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("El benchmark debe contener una sola versión por solicitud")
    if len({normalized(r["solicitud"]) for r in requests}) != len(requests):
        raise ValueError("El benchmark contiene solicitudes literales duplicadas")
    counts = Counter(r["codigo_oa"] for r in requests)
    if len(requests) != 295 or set(counts) != set(by_code) or set(counts.values()) != {5}:
        raise ValueError("Benchmark final: se requieren 295 solicitudes, cinco por cada OA")
    for row in requests:
        oa = by_code[row["codigo_oa"]]
        if row["curso"] != oa["curso"] or row["texto_oa"] != oa["texto_oa"]:
            raise ValueError(f"{row['id']}: curso o texto OA no coincide con el registro")


def validate_development(development, benchmark):
    ids = {r["id"] for r in benchmark}
    texts = {normalized(r["solicitud"]) for r in benchmark}
    if any(r["id"] in ids or normalized(r["solicitud"]) in texts for r in development):
        raise ValueError("Desarrollo y benchmark deben estar separados")
