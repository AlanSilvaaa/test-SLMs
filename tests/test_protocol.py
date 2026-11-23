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
    (tmp_path / "model-id").mkdir()
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


@pytest.mark.parametrize("proposed", ["SECRET_SOLUTION", None, ""])
def test_judge_stages_are_blind_and_solution_is_separate(tmp_path, monkeypatch, proposed):
    from common.contracts import Experiment
    from common.files import write_json
    from judge.evaluate import evaluate

    rubric = read_json(Path(__file__).parent.parent / "configs/rubric.v0.3-draft.json")
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
            "phase": "development",
            "problem": {
                "enunciado": "¿Cuánto es 2 + 3?",
                "solucion": proposed,
                "descripcion_figura": "FIGURA_DESCRITA",
            },
            "error": None,
        },
    )
    calls = []

    class FakeJudge:
        def __init__(self, cfg, run):
            pass

        def start(self):
            pass

        def metadata(self):
            return {"version": "fake"}

        def preflight(self, messages):
            return 1

        def close(self):
            pass

        def batch(self, messages, schema, **kwargs):
            assert len(messages) == 1
            text = json.dumps(messages[0])
            calls.append(text)
            if len(calls) == 1:
                value = {key: {"score": 1, "reason": ""} for key in schema["required"]}
            else:
                value = {
                    "solucion_independiente": "2 + 3 = 5",
                    "solution_accuracy": {
                        "score": 0,
                        "reason": "La propuesta no contiene procedimiento",
                    },
                }
            return [{"raw_response": json.dumps(value), "finish_reason": "stop"}]

    monkeypatch.setattr("judge.evaluate.LocalBench", FakeJudge)
    monkeypatch.setattr("judge.evaluate.validate_model", lambda *args, **kwargs: None)
    cfg = Experiment(
        model={"path": "test", "revision": "abc"},
        benchmark="unused",
        phase="development",
        output=str(tmp_path / "output"),
    )
    run = evaluate(cfg, source, rubric_path)
    record = load_rows(run / "evaluations.jsonl")[0]
    assert record["joint_pass"] is True
    assert record["solution"]["score"] == 0
    assert len(calls) == 2
    assert all("SECRET_MODEL" not in text for text in calls)
    assert all("FIGURA_DESCRITA" in text for text in calls)
    assert all("SECRET_SOLUTION" not in text for text in calls[:1])
    if proposed:
        assert "SECRET_SOLUTION" in calls[1]
    else:
        assert record["solution"]["reason"] == "Solución propuesta ausente"


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


def test_different_versions_of_same_candidate_cannot_cross_splits():
    original = example()
    revised = {**original, "version": 2, "enunciado": "Una reformulación con otros datos"}
    with pytest.raises(ValueError, match="IDs o versiones"):
        validate_separation([original], [revised], [request()])
    with pytest.raises(ValueError, match="IDs o versiones"):
        validate_separation([original, revised], [], [request()])


def test_generated_solution_may_be_missing_but_training_solution_may_not():
    from common.contracts import GeneratedProblem

    for solution in (None, ""):
        assert GeneratedProblem(enunciado="Un enunciado", solucion=solution).enunciado
        with pytest.raises(ValueError):
            Example.model_validate({**example(), "solucion": solution})


def test_report_keeps_request_compliance_separate_and_missing_visible(tmp_path):
    from common.contracts import DIMENSIONS
    from judge.report import summarize

    source, output = tmp_path / "input.jsonl", tmp_path / "summary.json"
    statement = {k: {"score": 1, "reason": ""} for k in (*DIMENSIONS, "request_compliance")}
    append_jsonl(
        source,
        {"statement": statement, "joint_pass": True, "solution": {"score": 0}, "error": None},
    )
    append_jsonl(
        source, {"statement": None, "joint_pass": None, "solution": None, "error": "parse error"}
    )
    summarize(source, output)
    result = read_json(output)
    assert set(result["dimensions"]) == set(DIMENSIONS)
    assert result["request_compliance"] == {"passed": 1, "denominator": 2, "missing": 1}
    assert result["joint_pass"] == {"passed": 1, "denominator": 2, "missing": 1}
    assert result["solution_accuracy"]["denominator"] == 1


def test_development_cannot_reuse_reserved_request():
    from common.data import validate_development

    with pytest.raises(ValueError, match="separados"):
        validate_development([{**request(), "version": 2}], [request()])


@pytest.fixture
def benchmark_v2(tmp_path):
    from common.data import AXES, V2_DISTRIBUTION
    from common.files import write_json

    path = tmp_path / "benchmark.jsonl"
    registry = tmp_path / "oa.json"
    objectives, requests = [], []
    for course, counts in enumerate(V2_DISTRIBUTION, 1):
        number = 0
        for (axis, code), count in zip(AXES.items(), counts, strict=True):
            for _ in range(count):
                number += 1
                if course == 6 and number == 15:
                    number += 1
                oa = f"MA{course:02d} OA {number:02d}"
                objectives.append(dict(codigo_oa=oa, curso=str(course), eje=code, texto_oa="OA"))
                for variant in range(1, 6):
                    key = f"BM2-{oa.replace(' ', '-')}-{variant:02d}"
                    row = request(key, oa)
                    del row["id"]
                    row.update(
                        id_solicitud=key,
                        version="2.1-candidata",
                        curso=f"{course}° básico",
                        texto_oa="OA",
                        eje=axis,
                        variante=variant,
                        tipo_representacion="Figura especificada",
                        especificacion_representacion="Figura a escala",
                        estado_revision="Revisada internamente",
                        observaciones_revision="",
                        cobertura_banco="Sin ejemplos",
                        solicitud=f"Solicitud {key}. Devuelve enunciado, descripcion_figura y solucion.",
                    )
                    requests.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
    write_json(registry, {"status": "approved", "version": "1", "objectives": objectives})
    return path, registry, requests


def test_final_benchmark_does_not_accept_one_request(benchmark_v2, tmp_path):
    from common.contracts import Experiment
    from common.generation import build_requests

    benchmark, registry, rows = benchmark_v2
    development = tmp_path / "development.jsonl"
    benchmark.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    append_jsonl(development, request("development"))
    cfg = Experiment(
        model={"path": "fake", "revision": "abc"},
        benchmark=str(benchmark),
        development=str(development),
        oa_registry=str(registry),
    )
    with pytest.raises(ValueError, match="725"):
        build_requests(cfg, "base_model", 0)


def test_benchmark_base_checks_full_candidate_without_runtime(benchmark_v2, monkeypatch):
    benchmark, _, _ = benchmark_v2
    monkeypatch.setattr("common.quick.check_inference_runtime", lambda: pytest.fail("runtime"))
    result = CliRunner().invoke(
        app, ["benchmark-base", "--benchmark", str(benchmark), "--check-only"]
    )
    assert result.exit_code == 0, result.output
    assert "2.1-candidata validado: 725 solicitudes" in result.output


def test_v2_base_preserves_prompt_and_figure_metadata(benchmark_v2, tmp_path):
    from common.contracts import Experiment, GeneratedProblem
    from common.excel import result_values
    from common.generation import build_requests

    benchmark, registry, expected = benchmark_v2
    development = tmp_path / "development.jsonl"
    append_jsonl(development, request("development"))
    cfg = Experiment(
        model={"path": "fake", "revision": "abc"},
        benchmark=str(benchmark),
        development=str(development),
        oa_registry=str(registry),
        load_conditions="Prueba sin pesos",
    )
    built, inputs = build_requests(cfg, "base_model", 0)
    assert len(built) == 725 and benchmark in inputs and registry in inputs
    assert built[0]["request"]["id"] == expected[0]["id_solicitud"]
    assert built[0]["request"]["version"] == expected[0]["version"]
    assert built[0]["messages"] == [{"role": "user", "content": expected[0]["solicitud"]}]
    problem = GeneratedProblem(enunciado="Texto", solucion="Solución", descripcion_figura="Escala")
    row = {**built[0], "problem": problem.model_dump()}
    assert result_values(row, 1, {})["descripcion_figura"] == "Escala"
    row["problem"]["descripcion_figura"] = ""
    assert result_values(row, 1, {})["state"] == "Descripción de figura ausente"


def test_smollm3_smoke_builds_exactly_five_base_requests_without_benchmark(tmp_path):
    from common.contracts import Experiment
    from common.generation import build_requests

    root = Path(__file__).resolve().parent.parent
    values = read_json(root / "configs/smollm3-smoke.example.json")
    values["development"] = str(root / values["development"])
    values["benchmark"] = str(tmp_path / "not-yet-created.jsonl")
    cfg = Experiment.model_validate(values)
    built, inputs = build_requests(cfg, "base_model", 0)
    assert cfg.repetitions == 1
    assert cfg.model.id == "HuggingFaceTB/SmolLM3-3B"
    assert len(built) == 5
    assert len(inputs) == 1
    assert all(not row["benchmark_separation_checked"] for row in built)
    assert all(len(row["messages"]) == 2 for row in built)
    # The relaxation applies only to development, not final benchmark runs.
    cfg.phase = "benchmark"
    with pytest.raises(FileNotFoundError):
        build_requests(cfg, "base_model", 0)


def test_smoke_still_checks_separation_if_reserved_benchmark_exists(tmp_path):
    from common.contracts import Experiment
    from common.generation import build_requests

    development, benchmark = tmp_path / "development.jsonl", tmp_path / "benchmark.jsonl"
    append_jsonl(development, request())
    append_jsonl(benchmark, request())
    cfg = Experiment(
        model={"path": "fake", "revision": "abc"},
        phase="development",
        development=str(development),
        benchmark=str(benchmark),
        repetitions=1,
    )
    with pytest.raises(ValueError, match="separados"):
        build_requests(cfg, "base_model", 0)


def test_relative_model_path_is_resolved_from_config_project_root(tmp_path, monkeypatch):
    from common.config import load_config
    from common.files import write_json

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    model = root / "models" / "snapshot"
    model.mkdir(parents=True)
    config = root / "configs" / "experiment.json"
    write_json(
        config,
        {
            "model": {"path": "models/snapshot", "revision": "abc"},
            "benchmark": "data/benchmark.jsonl",
        },
    )
    monkeypatch.chdir(tmp_path)
    assert load_config(config).model.path == str(model)
