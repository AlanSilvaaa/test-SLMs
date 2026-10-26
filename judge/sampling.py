"""Reproducible blind review samples. Pedagogical review remains human."""

import random
from collections import defaultdict

from common.files import load_rows, read_json, write_json


def sample_generations(rows):
    rng = random.Random(42)
    by_model = defaultdict(list)
    for row in rows:
        if row.get("phase") != "benchmark":
            raise ValueError("La muestra final requiere generaciones de benchmark")
        model = row["model"].get("id") or row["model"]["path"]
        by_model.setdefault(model, [])
        if row.get("problem") and not row.get("error"):
            by_model[model].append(row)
    if not by_model:
        raise ValueError("No hay enunciados para revisión docente")
    selected = []
    for model, group in sorted(by_model.items()):
        rng.shuffle(group)
        if len(group) < 30:
            raise ValueError(f"{model}: se requieren al menos 30 problemas")

        # Cover every OA, course and strategy present in the supplied runs.
        def strata(row):
            return {
                ("oa", row["request"]["codigo_oa"]),
                ("curso", row["request"]["curso"]),
                ("estrategia", row["module"], row.get("shots", 0)),
            }

        remaining = set().union(*(strata(r) for r in group))
        chosen = []
        while remaining:
            index = max(range(len(group)), key=lambda i: len(strata(group[i]) & remaining))
            row = group.pop(index)
            chosen.append(row)
            remaining -= strata(row)
        # Balance the remaining sample across model/strategy strata.
        buckets = defaultdict(list)
        for row in group:
            buckets[(row["module"], row.get("shots", 0))].append(row)
        while len(chosen) < 30:
            for key in sorted(buckets):
                if buckets[key] and len(chosen) < 30:
                    chosen.append(buckets[key].pop())
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def sample_dataset(train, validation, demonstrations, output):
    """Two per OA: one randomly among demos, one among all remaining examples."""
    rows = load_rows(train) + load_rows(validation)
    demos = read_json(demonstrations)
    groups = defaultdict(list)
    for row in rows:
        groups[row["codigo_oa"]].append(row)
    if len(groups) != 59:
        raise ValueError("Se requieren los 59 OA para la muestra de 118 problemas")
    rng, selected = random.Random(42), []
    for oa, group in sorted(groups.items()):
        keys = {(r["id"], r["version"]) for r in demos[oa]}
        examples = [r for r in group if (r["id"], r["version"]) in keys]
        others = [r for r in group if (r["id"], r["version"]) not in keys]
        if len(group) != 40 or len(examples) != 4 or not others:
            raise ValueError(f"{oa}: selección de ejemplos incompleta")
        for row in (rng.choice(examples), rng.choice(others)):
            selected.append(
                {
                    "request": {
                        k: row[k]
                        for k in (
                            "id",
                            "version",
                            "curso",
                            "codigo_oa",
                            "texto_oa",
                            "habilidad_evaluada",
                            "condiciones_curriculares",
                            "contexto",
                            "solicitud",
                        )
                    },
                    "problem": {k: row[k] for k in ("enunciado", "solucion")},
                    "model": {"id": "dataset"},
                    "module": "dataset",
                    "repetition": 0,
                }
            )
    # Source mapping retained alongside blind forms; never expose to the teacher.
    from common.files import append_jsonl
    from judge.report import export_human

    output.mkdir(parents=True, exist_ok=False)
    source = output / "private_dataset_sample.jsonl"
    for row in selected:
        append_jsonl(source, row)
    export_human(source, output / "forms")
    write_json(
        output / "sampling.json",
        {"seed": 42, "total": 118, "train": str(train), "validation": str(validation)},
    )
