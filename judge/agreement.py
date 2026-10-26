"""Agreement on unchanged annotations, with explicit missing comparisons."""

import csv
from pathlib import Path

from common.contracts import DIMENSIONS
from common.files import load_rows, write_json
from judge.evaluate import parse_verdicts


def compare(judge_sources, human_source, mapping_source, output):
    import json

    with Path(mapping_source).open(encoding="utf-8", newline="") as handle:
        mapping_rows = list(csv.DictReader(handle))
    mapping = {r["blind_id"]: r for r in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("IDs ciegos duplicados en el mapa")
    judgments = {}
    for path in judge_sources:
        for row in load_rows(path):
            key = (str(Path(row["generation_source"]).resolve()), row["generation_index"])
            if key in judgments:
                raise ValueError("Evaluaciones automáticas duplicadas")
            judgments[key] = row
    comparisons, missing, seen = [], [], set()
    for human in load_rows(human_source):
        blind_id = human["blind_id"]
        if blind_id in seen:
            raise ValueError("Usa una comparación independiente por evaluador")
        seen.add(blind_id)
        if not human.get("evaluador") or not human.get("rubric_version"):
            raise ValueError("Falta evaluador o versión de rúbrica")
        entry = mapping[blind_id]
        key = (str(Path(entry["source_file"]).resolve()), int(entry["generation_index"]))
        judge = judgments.get(key)
        if not judge or not judge.get("statement"):
            missing.append(blind_id)
            continue
        if human["rubric_version"] != judge["rubric_version"]:
            raise ValueError("Ambos evaluadores deben usar la misma rúbrica")
        scores = parse_verdicts(json.dumps(human["statement"]), DIMENSIONS)
        per_dimension = {
            k: scores[k]["score"] == judge["statement"][k]["score"] for k in DIMENSIONS
        }
        both_resolvable = (
            scores["solvability"]["score"] == 1 and judge["statement"]["solvability"]["score"] == 1
        )
        solution_agreement = None
        if both_resolvable and judge.get("solution") and human.get("solution"):
            solution = parse_verdicts(json.dumps({"solution": human["solution"]}), ["solution"])
            solution_agreement = solution["solution"]["score"] == judge["solution"]["score"]
        comparisons.append(
            {
                "blind_id": blind_id,
                "model": entry["model"],
                "module": entry["module"],
                "shots": entry["shots"],
                "dimensions": per_dimension,
                "joint_pass": all(scores[k]["score"] == 1 for k in DIMENSIONS)
                == judge["joint_pass"],
                "both_resolvable": both_resolvable,
                "solution_accuracy": solution_agreement,
            }
        )

    def rate(values):
        assessed = [v for v in values if v is not None]
        return {
            "agreed": sum(assessed),
            "denominator": len(assessed),
            "missing": len(values) - len(assessed),
            "percent": 100 * sum(assessed) / len(assessed) if assessed else None,
        }

    def summary(group):
        return {
            "dimensions": {k: rate([r["dimensions"][k] for r in group]) for k in DIMENSIONS},
            "joint_pass": rate([r["joint_pass"] for r in group]),
            "solution_accuracy": rate(
                [r["solution_accuracy"] for r in group if r["both_resolvable"]]
            ),
        }

    keys = sorted({(r["model"], r["module"], r["shots"]) for r in comparisons})
    result = {
        "overall": summary(comparisons),
        "by_model_strategy": {
            str(key): summary(
                [r for r in comparisons if (r["model"], r["module"], r["shots"]) == key]
            )
            for key in keys
        },
        "missing_judgments": missing,
        "missing_human_annotations": sorted(set(mapping) - seen),
        "disagreements": [
            r
            for r in comparisons
            if not all(r["dimensions"].values())
            or not r["joint_pass"]
            or r["solution_accuracy"] is False
        ],
    }
    # Refuse to replace earlier disagreement records or human annotations.
    with Path(output).open("x", encoding="utf-8"):
        pass
    write_json(output, result)
