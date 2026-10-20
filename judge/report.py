"""Keep missing judgments visible and preserve the protocol's denominators."""

from common.files import load_rows, write_json


def summarize(source, output):
    rows = load_rows(source)
    total = len(rows)
    criteria = sorted({k for r in rows if r.get("statement") for k in r["statement"]})
    result = {
        "total_generations": total,
        "evaluation_errors": sum(bool(r["error"]) for r in rows),
        "joint_pass": {"passed": sum(r["joint_pass"] is True for r in rows), "denominator": total},
        "dimensions": {},
    }
    for criterion in criteria:
        assessed = [r["statement"][criterion]["score"] for r in rows if r.get("statement")]
        result["dimensions"][criterion] = {
            "passed": sum(assessed),
            "denominator": total,
            "missing": total - len(assessed),
        }
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


def export_human(source, output):
    """Blind statement form. Solution review is a separate file opened afterwards."""
    import csv

    rows = load_rows(source)
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
                        "enunciado": row.get("problem", {}).get("enunciado", ""),
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
                        "solucion": row.get("problem", {}).get("solucion", ""),
                        "solucion_independiente": "",
                        "solution_accuracy": "",
                        "incumplimientos": "",
                    }
                )
            else:
                records.append(
                    {
                        "blind_id": key,
                        "generation_index": index,
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
