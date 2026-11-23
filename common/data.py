"""Fixed OA-stratified split and stable demonstration assignments."""

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common.contracts import BenchmarkV2Request, Example, Request
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


AXES = {
    "Números y operaciones": "numeros",
    "Patrones y álgebra": "patrones",
    "Geometría": "geometria",
    "Medición": "medicion",
    "Datos y probabilidades": "datos",
}
V2_DISTRIBUTION = [
    (10, 2, 3, 3, 2),
    (11, 2, 3, 3, 3),
    (11, 2, 5, 4, 4),
    (12, 2, 5, 5, 3),
    (13, 2, 3, 4, 5),
    (8, 3, 5, 4, 3),
]
PILOT_DISTRIBUTION = [
    (3, 0, 0, 1, 0),
    (4, 0, 0, 1, 0),
    (6, 0, 0, 2, 0),
    (9, 1, 0, 3, 0),
    (12, 1, 0, 2, 3),
    (8, 1, 0, 2, 0),
]
V2_REQUESTS = 725


def load_benchmark(path, *, require_v2=False):
    rows = load_rows(path)
    if require_v2 or any("id_solicitud" in row for row in rows):
        return [BenchmarkV2Request.model_validate(row).model_dump() for row in rows]
    return load_rows(path, Request)


def validate_coverage(objectives, distribution):
    expected = Counter(
        {
            (str(course), axis): count
            for course, counts in enumerate(distribution, 1)
            for axis, count in zip(AXES.values(), counts, strict=True)
            if count
        }
    )
    actual = Counter((r["curso"], r["eje"]) for r in objectives)
    if actual != expected:
        raise ValueError("La distribución curso/eje no coincide con la tabla de la memoria")


def validate_request_counts(requests, codes, expected):
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("El benchmark debe contener una sola versión por solicitud")
    if len({normalized(r["solicitud"]) for r in requests}) != len(requests):
        raise ValueError("El benchmark contiene solicitudes literales duplicadas")
    counts = Counter(r["codigo_oa"] for r in requests)
    if len(requests) != expected or set(counts) != set(codes) or set(counts.values()) != {5}:
        raise ValueError(f"Se requieren {expected} solicitudes, cinco por cada OA")


def validate_benchmark_v2(requests):
    objectives = {}
    if len({r["version"] for r in requests}) != 1:
        raise ValueError("Todas las solicitudes deben pertenecer a la misma versión")
    for row in requests:
        code, variant = row["codigo_oa"], row["variante"]
        course = row["curso"]
        if (
            not re.fullmatch(rf"MA0{course} OA [0-9]{{2}}", code)
            or not 1 <= int(code[-2:]) <= (20, 22, 26, 27, 27, 24)[int(course) - 1]
            or code == "MA06 OA 15"
            or row["id"] != f"BM2-{code.replace(' ', '-')}-{variant:02d}"
        ):
            raise ValueError(f"{row['id']}: identificador, OA o curso inválido")
        oa = {"curso": course, "eje": AXES[row["eje"]], "texto_oa": row["texto_oa"]}
        if objectives.setdefault(code, oa) != oa:
            raise ValueError(f"{code}: metadatos curriculares inconsistentes")
    validate_request_counts(requests, objectives, V2_REQUESTS)
    validate_coverage(objectives.values(), V2_DISTRIBUTION)


def validate_requests(requests, registry_path, *, legacy=False, require_approved=True):
    """Compare final requests against the approved register, independently of the file."""
    expected = 295 if legacy else V2_REQUESTS
    registry = read_json(registry_path)
    if (require_approved and registry.get("status") != "approved") or not registry.get("version"):
        raise ValueError("El registro de OA debe estar revisado, aprobado y versionado")
    objectives = registry["objectives"]
    by_code = {r["codigo_oa"]: r for r in objectives}
    if len(objectives) != expected // 5 or len(by_code) != expected // 5:
        raise ValueError(f"El registro requiere los {expected // 5} OA seleccionados")
    validate_coverage(objectives, PILOT_DISTRIBUTION if legacy else V2_DISTRIBUTION)
    validate_request_counts(requests, by_code, expected)
    if not legacy:
        validate_benchmark_v2(requests)
    for row in requests:
        oa = by_code[row["codigo_oa"]]
        if (
            row["curso"] != oa["curso"]
            or row["texto_oa"] != oa["texto_oa"]
            or (not legacy and AXES[row["eje"]] != oa["eje"])
        ):
            raise ValueError(f"{row['id']}: curso, eje o texto OA no coincide con el registro")


def validate_development(development, benchmark):
    ids = {r["id"] for r in benchmark}
    texts = {normalized(r["solicitud"]) for r in benchmark}
    if any(r["id"] in ids or normalized(r["solicitud"]) in texts for r in development):
        raise ValueError("Desarrollo y benchmark deben estar separados")
