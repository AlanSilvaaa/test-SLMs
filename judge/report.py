"""Keep missing judgments visible and preserve the protocol's denominators."""

from common.contracts import DIMENSIONS
from common.files import load_rows, write_json


def summarize(source, output):
    rows = load_rows(source)
    versions = {r.get("rubric_version") for r in rows}
    if len(versions) > 1:
        raise ValueError("No se pueden mezclar versiones de rúbrica en un resumen")
    total = len(rows)
    criteria = DIMENSIONS
    result = {
        "total_generations": total,
        "evaluation_errors": sum(bool(r["error"]) for r in rows),
        "joint_pass": {
            "passed": sum(r["joint_pass"] is True for r in rows),
            "denominator": total,
            "missing": sum(r["joint_pass"] is None for r in rows),
        },
        "dimensions": {},
    }
    for criterion in criteria:
        assessed = [r["statement"][criterion]["score"] for r in rows if r.get("statement")]
        result["dimensions"][criterion] = {
            "passed": sum(assessed),
            "denominator": total,
            "missing": total - len(assessed),
        }
    compliance = [r["statement"]["request_compliance"]["score"] for r in rows if r.get("statement")]
    result["request_compliance"] = {
        "passed": sum(compliance),
        "denominator": total,
        "missing": total - len(compliance),
    }
    result["by_model_strategy"] = {}
    for row in rows:
        key = str((row.get("generator"), row.get("module"), row.get("shots")))
        group = result["by_model_strategy"].setdefault(key, {"total": 0, "joint_passed": 0})
        group["total"] += 1
        group["joint_passed"] += row["joint_pass"] is True
    resolvable = [
        r for r in rows if r.get("statement", {}) and r["statement"]["solvability"]["score"] == 1
    ]
    result["solution_accuracy"] = {
        "passed": sum(
            r.get("solution") is not None and r["solution"]["score"] == 1 for r in resolvable
        ),
        "denominator": len(resolvable),
        "missing": sum(r.get("solution") is None for r in resolvable),
    }
    write_json(output, result)


def export_human(source, output, *, sample=False):
    """Blind statement form. Solution review is a separate file opened afterwards."""
    import csv

    from judge.sampling import sample_generations

    sources = source if isinstance(source, list) else [source]
    rows = []
    for path in sources:
        for index, row in enumerate(load_rows(path)):
            rows.append({**row, "source_file": str(path), "source_index": index})
    if sample:
        rows = sample_generations(rows)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("statements", "solutions", "private_mapping"):
        records = []
        for index, row in enumerate(rows):
            key = f"H{index + 1:06d}"
            if name == "statements":
                records.append(
                    {
                        "blind_id": key,
                        **row["request"],
                        "enunciado": (row.get("problem") or {}).get("enunciado", ""),
                        "evaluador": "",
                        "rubric_version": "",
                        "solvability": "",
                        "linguistic_clarity": "",
                        "contextual_coherence": "",
                        "pedagogical_appropriateness": "",
                        "curriculum_alignment": "",
                        "request_compliance": "",
                        "incumplimientos": "",
                    }
                )
            elif name == "solutions":
                records.append(
                    {
                        "blind_id": key,
                        "solucion": (row.get("problem") or {}).get("solucion", ""),
                        "solucion_independiente": "",
                        "solution_accuracy": "",
                        "incumplimientos": "",
                    }
                )
            else:
                records.append(
                    {
                        "blind_id": key,
                        "generation_index": row["source_index"],
                        "source_file": row["source_file"],
                        "model": str(row.get("model")),
                        "module": row.get("module"),
                        "shots": row.get("shots"),
                        "repetition": row.get("repetition"),
                    }
                )
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
