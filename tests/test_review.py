import csv
import json

import pytest

from common.contracts import DIMENSIONS
from common.files import append_jsonl, load_rows, read_json, write_json
from judge.agreement import compare
from judge.sampling import sample_dataset, sample_generations
from tests.test_protocol import example


def test_dataset_sample_has_one_demo_and_one_other_per_oa(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    demonstrations, assignments = tmp_path / "demos.json", {}
    for oa in range(59):
        key = f"oa{oa}"
        assignments[key] = [{"id": f"e{oa}-{i}", "version": 1} for i in range(4)]
        for i in range(40):
            append_jsonl(train if i < 34 else validation, example(f"e{oa}-{i}", key))
    write_json(demonstrations, assignments)
    output = tmp_path / "review"
    sample_dataset(train, validation, demonstrations, output)
    selected = load_rows(output / "private_dataset_sample.jsonl")
    assert len(selected) == 118
    for oa, demos in assignments.items():
        pair = [r for r in selected if r["request"]["codigo_oa"] == oa]
        assert len(pair) == 2
        assert sum(r["request"]["id"] in {d["id"] for d in demos} for r in pair) == 1
    with (output / "forms/statements.csv").open() as handle:
        assert "solucion" not in next(csv.reader(handle))


def generation(index, model="model"):
    return {
        "phase": "benchmark",
        "model": {"id": model},
        "module": "few-shot" if index % 2 else "base_model",
        "shots": index % 2,
        "request": {"codigo_oa": f"oa{index % 7}", "curso": str(index % 6 + 1)},
        "problem": {"enunciado": f"Problema {index}"},
        "error": None,
    }


def test_generation_sample_is_reproducible_and_covers_strata():
    rows = [generation(i, model) for model in ("A", "B") for i in range(100)]
    selected = sample_generations(rows)
    assert selected == sample_generations(rows)
    for model in ("A", "B"):
        group = [r for r in selected if r["model"]["id"] == model]
        assert len(group) == 30
        assert len({r["request"]["codigo_oa"] for r in group}) == 7
        assert len({r["request"]["curso"] for r in group}) == 6
        assert len({r["module"] for r in group}) == 2
    with pytest.raises(ValueError, match="30"):
        sample_generations([generation(0)])


def scores(solvable=True):
    return {
        k: {
            "score": 1 if k != "solvability" or solvable else 0,
            "reason": "" if k != "solvability" or solvable else "Faltan datos",
        }
        for k in DIMENSIONS
    }


def test_agreement_uses_jointly_resolvable_denominator_and_preserves_annotations(tmp_path):
    source = tmp_path / "generations.jsonl"
    judge, human, mapping, output = [
        tmp_path / name for name in ("judge.jsonl", "human.jsonl", "map.csv", "agreement.json")
    ]
    with mapping.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["blind_id", "source_file", "generation_index", "model", "module", "shots"],
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "blind_id": str(index),
                    "source_file": str(source),
                    "generation_index": index,
                    "model": "M",
                    "module": "base_model",
                    "shots": 0,
                }
            )
            append_jsonl(
                judge,
                {
                    "generation_source": str(source),
                    "generation_index": index,
                    "rubric_version": "1",
                    "statement": scores(),
                    "joint_pass": True,
                    "solution": {"score": 1, "reason": ""},
                },
            )
            append_jsonl(
                human,
                {
                    "blind_id": str(index),
                    "evaluador": "docente",
                    "rubric_version": "1",
                    "statement": scores(solvable=index == 0),
                    "solution": {"score": 0, "reason": "Respuesta incorrecta"},
                },
            )
    originals = [p.read_bytes() for p in (judge, human, mapping)]
    compare([judge], human, mapping, output)
    result = read_json(output)
    assert result["overall"]["solution_accuracy"]["denominator"] == 1
    assert result["overall"]["dimensions"]["solvability"]["percent"] == 50
    assert len(result["disagreements"]) == 2
    assert originals == [p.read_bytes() for p in (judge, human, mapping)]
    with pytest.raises(FileExistsError):
        compare([judge], human, mapping, output)


def test_performance_keeps_native_throughput_separate_from_decode(tmp_path):
    from common.performance import summarize_performance

    source, output = tmp_path / "generations.jsonl", tmp_path / "summary.json"
    for repetition, rate in enumerate([10, 20]):
        native = tmp_path / f"native{repetition}.json"
        write_json(native, {"output_throughput": rate, "duration": 2, "completed": 1, "failed": 0})
        append_jsonl(
            source,
            {
                "repetition": repetition,
                "bench_result": str(native),
                "error": None,
                "decode_tokens_per_second": 100,
                "ttft_s": 0.2,
                "latency_s": 1.2,
                "decode_s": 1,
                "tpot_s": 0.01,
                "output_limit_reached": False,
            },
        )
    summarize_performance(source, output)
    result = json.loads(output.read_text())
    assert result["aggregate_output_tokens_per_second"]["mean"] == 15
    assert result["aggregate_output_tokens_per_second"]["std"] == 5
    assert result["per_problem"]["decode_tokens_per_second"]["mean"] == 100
