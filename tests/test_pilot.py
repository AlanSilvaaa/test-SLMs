"""Pilot preparation and orchestration with synthetic records and fake models."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from common.contracts import Experiment, Model
from common.data import validate_requests
from common.files import load_rows, read_json
from common.generation import build_requests
from common.pilot import run_pilot
from common.pilot_data import AXES, prepare_pilot, validate_pilot
from main import app


@pytest.fixture
def pilot_package(tmp_path, monkeypatch):
    distribution = {
        "1": {"numeros": 3, "medicion": 1},
        "2": {"numeros": 4, "medicion": 1},
        "3": {"numeros": 6, "medicion": 2},
        "4": {"numeros": 9, "medicion": 3, "patrones": 1},
        "5": {"numeros": 12, "medicion": 2, "patrones": 1, "datos": 3},
        "6": {"numeros": 8, "medicion": 2, "patrones": 1},
    }
    objectives, candidates, requests = [], [], []
    axis_names = {v: k for k, v in AXES.items()}
    for course, axes in distribution.items():
        for axis, number in axes.items():
            for i in range(number):
                code = f"{course}-{axis}-{i}"
                text = f"Texto curricular sintético {code}."
                objectives.append(
                    {
                        "curso": f"{course}° básico",
                        "codigo_oa": code,
                        "eje": axis_names[axis],
                        "texto_original": text,
                        "usable": "si",
                    }
                )
                common = {
                    "version": 1,
                    "curso": f"{course}° básico",
                    "codigo_oa": code,
                    "habilidad_evaluada": "sumar",
                    "condiciones_curriculares": "Hasta 10",
                    "contexto": "huerto",
                }
                candidates.append(
                    {**common, "id_problema": f"DS-{code}", "solicitud": f"Dataset: {text}"}
                )
                for j in range(5):
                    requests.append(
                        {
                            **common,
                            "id_solicitud": f"BM-{code}-{j}",
                            "solicitud": f"Benchmark {j}: {text}",
                        }
                    )

    class Sheet:
        def __init__(self, rows):
            keys = list(rows[0])
            self.values = [tuple(keys), *(tuple(row[k] for k in keys) for row in rows)]

    class Book(dict):
        def close(self):
            pass

    books = iter(
        [
            Book({"Dataset piloto": Sheet(candidates), "Benchmark piloto": Sheet(requests)}),
            Book({"OA": Sheet(objectives)}),
        ]
    )
    monkeypatch.setattr("openpyxl.load_workbook", lambda *args, **kwargs: next(books))
    source, oa_source = tmp_path / "pilot.xlsx", tmp_path / "oa.xlsx"
    source.write_bytes(b"synthetic workbook")
    oa_source.write_bytes(b"synthetic registry")
    return prepare_pilot(source, oa_source, tmp_path / "package")


def config(package):
    return Experiment(
        model=Model(id="HuggingFaceTB/SmolLM3-3B", path="missing", revision="a" * 40),
        phase="pilot",
        benchmark=str(package / "benchmark.jsonl"),
        oa_registry=str(package / "oa.json"),
        pilot_manifest=str(package / "manifest.json"),
        development="examples/smoke/requests.jsonl",
        repetitions=1,
    )


def test_import_preserves_draft_and_builds_all_295_without_accepting_training(pilot_package):
    cfg = config(pilot_package)
    inputs = validate_pilot(cfg)
    built, _ = build_requests(cfg, "base_model", 0)
    assert len(built) == 295
    assert len({r["request"]["codigo_oa"] for r in built}) == 59
    assert all("solucion" not in r["request"] for r in built)
    assert len(load_rows(pilot_package / "dataset-requests.jsonl")) == 59
    assert read_json(cfg.oa_registry)["status"] == "draft"
    assert Path(cfg.pilot_manifest) in inputs
    assert not (pilot_package / "train.jsonl").exists()
    with pytest.raises(ValueError, match="aprobado"):
        validate_requests(load_rows(cfg.benchmark), cfg.oa_registry)


def test_changed_package_rejected_before_runtime_or_download(pilot_package, monkeypatch):
    (pilot_package / "benchmark.jsonl").write_text("changed")
    monkeypatch.setattr("common.runtime.check_inference_runtime", lambda: pytest.fail("runtime"))
    with pytest.raises(ValueError, match="cambió"):
        run_pilot(data=pilot_package)


@pytest.mark.parametrize("changes", [{"request_limit": 5}, {"repetitions": 2}])
def test_pilot_cannot_silently_limit_or_repeat(pilot_package, changes):
    cfg = config(pilot_package).model_copy(update=changes)
    with pytest.raises(ValueError, match="una pasada"):
        build_requests(cfg, "base_model", 0)


def test_nightly_sequence_continues_after_model_failure_and_records_status(
    pilot_package, tmp_path, monkeypatch
):
    from common.pilot_data import ROOT

    prepared, generated, reports = [], [], []
    monkeypatch.setattr("common.runtime.check_inference_runtime", lambda: None)
    monkeypatch.setattr("common.inference.check_local_port", lambda port: None)

    def fake_prepare(identifier, models_dir, **kwargs):
        prepared.append((identifier, kwargs))
        if len(prepared) == 2:
            raise RuntimeError("synthetic download failure")
        return Model(id=identifier, path=str(tmp_path), revision="a" * 40)

    def fake_infer(cfg, module, shots):
        generated.append(cfg.model_dump())
        assert module == "base_model" and shots == 0
        return tmp_path / cfg.model.id

    monkeypatch.setattr("common.pilot.prepare_model", fake_prepare)
    monkeypatch.setattr("common.generation.infer", fake_infer)
    with pytest.raises(RuntimeError, match="1 modelos fallidos"):
        run_pilot(
            pilot_package, tmp_path / "output", offline=True, port=8001, report=reports.append
        )
    catalog = read_json(ROOT / "configs/models.json")["generators"]
    assert [row[0] for row in prepared] == catalog
    assert all(row[1] == {"offline": True} for row in prepared)
    assert len(generated) == 5
    assert all(
        c["phase"] == "pilot"
        and c["repetitions"] == 1
        and c["port"] == 8001
        and c["request_limit"] is None
        for c in generated
    )
    state_path = next((tmp_path / "output").rglob("status.json"))
    state = read_json(state_path)
    assert state["status"] == "completed_with_errors"
    assert [r["status"] for r in state["models"]] == ["complete", "failed"] + ["complete"] * 4
    assert Path(state["models"][1]["log"]).is_file()


def test_nightly_interrupt_does_not_launch_remaining_models(pilot_package, tmp_path, monkeypatch):
    monkeypatch.setattr("common.runtime.check_inference_runtime", lambda: None)
    monkeypatch.setattr("common.inference.check_local_port", lambda port: None)
    monkeypatch.setattr(
        "common.pilot.prepare_model", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        run_pilot(pilot_package, tmp_path / "output")
    state = read_json(next((tmp_path / "output").rglob("status.json")))
    assert state["status"] == "interrupted"
    assert len(state["models"]) == 1
    assert state["models"][0]["status"] == "interrupted"


def test_pilot_cli_forwards_flags_without_ml(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("common.pilot.run_pilot", lambda *a, **kw: calls.append(a))
    result = CliRunner().invoke(
        app, ["pilot", "--data", str(tmp_path), "--offline", "--port", "8002"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0][0] == tmp_path and calls[0][3:] == (True, 8002)


def test_background_launch_detaches_and_reuses_current_python(pilot_package, tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from common.pilot import launch_pilot

    calls = []

    def fake_process(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("synthetic launch")
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("common.pilot.subprocess.Popen", fake_process)
    pid, log = launch_pilot(pilot_package, tmp_path / "out", tmp_path / "models", True, 8001)
    assert pid == 123 and log.read_text() == "synthetic launch"
    command, options = calls[0]
    assert command[0] == sys.executable
    assert "--offline" in command and "--background" not in command
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert options["env"]["PYTHONUNBUFFERED"] == "1"
