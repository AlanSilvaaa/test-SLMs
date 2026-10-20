import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from common.contracts import Example, Request, Rubric
from common.data import prepare, validate_separation
from common.files import append_jsonl, load_rows, read_json
from common.prompts import conversation
from few_shot.infer import messages
from judge.evaluate import parse_verdicts, statement_messages
from lora.train import encode_example
from main import app


def request(key="q1", oa="oa1"):
    return dict(
        id=key,
        version=1,
        curso="1 básico",
        codigo_oa=oa,
        texto_oa="Sumar",
        habilidad_evaluada="sumar",
        condiciones_curriculares="Hasta 10",
        contexto="huerto",
        solicitud=f"Solicitud {key}",
    )


def example(key="e1", oa="oa1"):
    return dict(
        **request(key, oa),
        enunciado=f"Enunciado {key}",
        solucion="3 + 2 = 5",
        procedencia="test sintético",
        estado="aceptado",
    )


def test_cli_help_does_not_import_ml():
    import sys

    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--module" in result.stdout
    assert "--server-pid" not in result.stdout
    assert "export-bench" not in result.stdout
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules


def test_no_training_for_prompt_strategies(tmp_path):
    # The action guard is tested with a real lightweight configuration.
    cfg = tmp_path / "configs" / "x.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps(
            {
                "model": {"path": "model-id", "revision": "abc"},
                "benchmark": "data/q.jsonl",
            }
        )
    )
    result = CliRunner().invoke(
        app, ["run", "--module", "skill", "--action", "train", "--config", str(cfg)]
    )
    assert result.exit_code != 0
    assert "Solo LoRA" in result.output


def test_rotation_is_nested_and_oa_specific():
    examples = {(f"e{i}", 1): example(f"e{i}") for i in range(4)}
    assignments = {"oa1": [{"id": f"e{i}", "version": 1} for i in range(4)]}
    for ordinal in range(5):
        for shots in range(1, 5):
            actual, meta = messages(
                request(), examples=examples, assignments=assignments, ordinal=ordinal, shots=shots
            )
            expected = [f"e{(ordinal + i) % 4}" for i in range(shots)]
            assert [r["id"] for r in meta["demonstrations"]] == expected
            assert len(actual) == 2 + 2 * shots
    examples[("e0", 1)]["codigo_oa"] = "other"
    with pytest.raises(ValueError, match="otro OA"):
        messages(request(), examples=examples, assignments=assignments, ordinal=0, shots=4)


def test_fixed_split_and_train_only_demonstrations(tmp_path):
    dataset, benchmark = tmp_path / "accepted.jsonl", tmp_path / "benchmark.jsonl"
    for oa in range(59):
        for i in range(40):
            append_jsonl(dataset, example(f"e-{oa}-{i}", f"oa-{oa}"))
        for i in range(5):
            append_jsonl(benchmark, request(f"q-{oa}-{i}", f"oa-{oa}"))
    a = prepare(dataset, benchmark, tmp_path / "output")
    b = prepare(dataset, benchmark, tmp_path / "output")
    train, validation = (
        load_rows(a / "train.jsonl", Example),
        load_rows(a / "validation.jsonl", Example),
    )
    assert len(train) == 2006 and len(validation) == 354
    for name in ("train.jsonl", "validation.jsonl", "demonstrations.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    keys = {(r["id"], r["version"]) for r in train}
    for demos in read_json(a / "demonstrations.json").values():
        assert len(demos) == 4
        assert all((r["id"], r["version"]) in keys for r in demos)


def test_leakage_and_duplicates_are_rejected():
    row = example()
    with pytest.raises(ValueError, match="solapamiento"):
        validate_separation([row], [row], [request()])
    second = example("e2")
    second["enunciado"] = row["enunciado"].upper() + "!"
    with pytest.raises(ValueError, match="duplicados"):
        validate_separation([row, second], [], [request()])
    reserved = request()
    row["solicitud"] = reserved["solicitud"]
    with pytest.raises(ValueError, match="benchmark"):
        validate_separation([row], [], [reserved])


def test_judge_does_not_receive_solution_or_generator():
    rubric = Rubric.model_validate(
        read_json(Path(__file__).parent.parent / "configs/rubric.draft.json")
    )
    raw = json.dumps(
        statement_messages(
            {**request(), "model": "SECRET_MODEL", "solucion": "SECRET_SOLUTION"},
            "Solo enunciado",
            rubric,
        )
    )
    assert "SECRET_MODEL" not in raw and "SECRET_SOLUTION" not in raw


@pytest.mark.parametrize(
    "verdict",
    [
        {"score": True, "reason": ""},
        {"score": 0, "reason": ""},
        {"score": 1, "reason": "extra"},
        {"score": 2, "reason": "bad"},
    ],
)
def test_strict_judge_validation(verdict):
    with pytest.raises(ValueError):
        parse_verdicts(json.dumps({"criterion": verdict}), ["criterion"])


def test_valid_judge_and_contracts():
    assert parse_verdicts('{"c":{"score":0,"reason":"Falta un dato"}}', ["c"])["c"]["score"] == 0
    Request.model_validate(request())
    Example.model_validate(example())
    assert len(conversation(request())) == 2


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        prefix = [1, 2, 3]
        return prefix + ([4, 5] if messages[-1]["role"] == "assistant" else [])


def test_response_mask_and_no_truncation():
    encoded = encode_example(FakeTokenizer(), example(), 5, {})
    assert encoded["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match="No se truncará"):
        encode_example(FakeTokenizer(), example(), 4, {})


def test_draft_judge_never_connects(tmp_path):
    from common.contracts import Experiment
    from judge.evaluate import evaluate

    cfg = Experiment(model={"path": "test", "revision": "abc"}, benchmark="unused")
    rubric = Path(__file__).parent.parent / "configs/rubric.draft.json"
    with pytest.raises(ValueError, match="draft"):
        evaluate(cfg, tmp_path / "missing.jsonl", rubric)


def test_judge_stages_are_blind_and_solution_is_separate(tmp_path, monkeypatch):
    from common.contracts import Experiment
    from common.files import write_json
    from judge.evaluate import evaluate

    rubric = read_json(Path(__file__).parent.parent / "configs/rubric.draft.json")
    rubric["status"] = "approved"
    rubric_path = tmp_path / "rubric.json"
    write_json(rubric_path, rubric)
    source = tmp_path / "generations.jsonl"
    append_jsonl(
        source,
        {
            "request": request(),
            "repetition": 0,
            "model": "SECRET_MODEL",
            "problem": {"enunciado": "¿Cuánto es 2 + 3?", "solucion": "SECRET_SOLUTION"},
            "error": None,
        },
    )
    calls = []

    class FakeJudge:
        def __init__(self, cfg):
            pass

        def metadata(self):
            return {"version": "fake"}

        def preflight(self, messages):
            return 1

        def close(self):
            pass

        def complete(self, messages, schema, **kwargs):
            text = json.dumps(messages)
            calls.append(text)
            if len(calls) == 1:
                value = {key: {"score": 1, "reason": ""} for key in schema["required"]}
            elif len(calls) == 2:
                value = {"solucion_independiente": "2 + 3 = 5"}
            else:
                value = {
                    "solution_accuracy": {
                        "score": 0,
                        "reason": "La propuesta no contiene procedimiento",
                    }
                }
            return {"raw_response": json.dumps(value), "finish_reason": "stop"}

    monkeypatch.setattr("judge.evaluate.LocalVLLM", FakeJudge)
    cfg = Experiment(
        model={"path": "test", "revision": "abc"},
        benchmark="unused",
        output=str(tmp_path / "output"),
    )
    run = evaluate(cfg, source, rubric_path)
    record = load_rows(run / "evaluations.jsonl")[0]
    assert record["joint_pass"] is True
    assert record["solution"]["score"] == 0
    assert len(calls) == 3
    assert all("SECRET_MODEL" not in text for text in calls)
    assert all("SECRET_SOLUTION" not in text for text in calls[:2])
    assert "SECRET_SOLUTION" in calls[2]


def test_lora_selection_tie_breaks_and_dataset_identity(tmp_path):
    from common.files import write_json
    from lora.train import select

    runs = []
    for index, rate in enumerate([0.00005, 0.0001, 0.0002]):
        run = tmp_path / str(index)
        adapter = run / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fake")
        write_json(
            run / "selection.json",
            {
                "learning_rate": rate,
                "eval_loss": 1.0,
                "epoch": 2 if index == 0 else 1,
                "adapter": str(adapter),
            },
        )
        write_json(
            run / "manifest.json",
            {"config": {"model": "same", "learning_rate": rate}, "inputs": {"train": "same-hash"}},
        )
        runs.append(run)
    target = tmp_path / "selected.json"
    select(runs, target)
    assert read_json(target)["learning_rate"] == 0.0001
    write_json(
        runs[2] / "manifest.json",
        {"config": {"model": "other", "learning_rate": 0.0002}, "inputs": {"train": "same-hash"}},
    )
    with pytest.raises(ValueError, match="no comparten"):
        select(runs, tmp_path / "invalid.json")
