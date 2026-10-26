"""Export synthetic observations only; no inference, judge or GPU is invoked."""

from datetime import datetime, timezone

import pytest
from openpyxl import load_workbook
from typer.testing import CliRunner

from common.excel import RESULT_COLUMNS, export_run
from common.files import append_jsonl, new_run, read_json, write_json
from main import app


def synthetic_run(folder, *, nested=False, rows=None):
    artifacts = folder / "detalles" if nested else folder
    artifacts.mkdir(parents=True)
    config = {
        "model": {"id": "HuggingFaceTB/SmolLM3-3B", "revision": "a" * 40},
        "module": "base_model",
        "phase": "development",
        "seed": 42,
        "repetitions": 1,
        "max_tokens": 1024,
    }
    write_json(
        artifacts / "manifest.json",
        {"created_utc": "20261026T192134Z", "config": config, "packages": {"vllm": "0.15.1+cu130"}},
    )
    if rows is None:
        rows = [
            {
                "request": {
                    "id": "synthetic-1",
                    "version": 1,
                    "curso": "1",
                    "codigo_oa": "DEV-SUMA",
                    "contexto": "huerto",
                    "solicitud": "Solicitud sintética",
                },
                "problem": {
                    "enunciado": "Hay 2 flores y plantan 3. ¿Cuántas flores hay?",
                    "solucion": "2 + 3 = 5 flores.",
                },
                "raw_response": "Respuesta sintética",
                "repetition": 0,
                "module": "base_model",
                "error": None,
                "parse_error": None,
                "output_limit_reached": False,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "latency_s": 1.2,
                "ttft_s": 0.2,
                "decode_tokens_per_second": 19.0,
            }
        ]
    for row in rows:
        append_jsonl(artifacts / "generations.jsonl", row)
    write_json(artifacts / "status.json", {"status": "complete"})
    write_json(
        artifacts / "performance.json",
        {
            "per_problem": {"latency_s": {"mean": 1.2, "median": 1.2, "std": 0.0, "n": 1}},
            "aggregate_output_tokens_per_second": {"mean": 10.0, "median": 10.0, "std": 0.0},
            "repetitions": {
                "0": {"aggregate_output_tokens_per_second": 10.0, "benchmark_duration_s": 2.0}
            },
        },
    )
    return artifacts


def result_map(book, row=2):
    sheet = book["Resultados"]
    return {
        sheet.cell(1, col).value: sheet.cell(row, col).value
        for col in range(1, len(RESULT_COLUMNS) + 1)
    }


@pytest.mark.parametrize("nested", [False, True])
def test_export_old_and_new_layout_preserves_records_and_numeric_types(tmp_path, nested):
    run = tmp_path / "run"
    artifacts = synthetic_run(run, nested=nested)
    original = (artifacts / "generations.jsonl").read_bytes()
    output = export_run(run)
    book = load_workbook(output)
    assert book.sheetnames == ["Resultados", "Resumen"]
    row = result_map(book)
    assert row["Enunciado"] == "Hay 2 flores y plantan 3. ¿Cuántas flores hay?"
    assert row["Solución"] == "2 + 3 = 5 flores."
    assert row["Tokens entrada"] == 10 and isinstance(row["Tokens entrada"], int)
    assert row["Latencia total (s)"] == 1.2
    assert row["TPOT (s/token)"] is None  # not replaced with zero
    assert row["Repetición (desde 1)"] == 1
    assert row["Estado técnico"] == "Sin errores técnicos"
    summary = {r[0].value: r[1].value for r in book["Resumen"].iter_rows(min_row=4)}
    assert summary["Fecha de ejecución"] == datetime(2026, 10, 26, 16, 21, 34)
    assert summary["Tasa agregada de salida: media"] == 10
    assert summary["Velocidad de decode: media"] is None  # no invented summary
    assert summary["Latencia total: desviación estándar poblacional"] == 0
    assert book["Resultados"].freeze_panes == "E2"
    assert "Problemas" in book["Resultados"].tables
    assert (artifacts / "generations.jsonl").read_bytes() == original
    book.close()
    with pytest.raises(FileExistsError):
        export_run(run)


def test_parse_errors_and_missing_solutions_are_exported_without_pedagogical_scores(tmp_path):
    rows = [
        {
            "request": {"id": "bad"},
            "problem": None,
            "error": None,
            "parse_error": "JSON inválido",
            "raw_response": "=1+2",
            "output_limit_reached": True,
        },
        {
            "request": {"id": "empty-solution"},
            "problem": {"enunciado": "¿Cuánto es 2+3?", "solucion": None},
            "error": None,
            "parse_error": None,
        },
    ]
    synthetic_run(tmp_path / "run", rows=rows)
    book = load_workbook(export_run(tmp_path / "run"))
    assert result_map(book)["Estado técnico"] == "Error de formato"
    assert result_map(book, 3)["Estado técnico"] == "Solución ausente"
    assert result_map(book)["Respuesta original"] == "=1+2"
    raw_column = [c[0] for c in RESULT_COLUMNS].index("Respuesta original") + 1
    assert book["Resultados"].cell(2, raw_column).data_type == "s"
    assert "Cumplimiento conjunto" not in result_map(book)
    assert result_map(book)["Enunciado"] is None
    book.close()


def test_long_text_is_preserved_in_full_and_control_characters_are_explicit(tmp_path):
    text = "Largo 😀 " * 8000
    rows = [
        {
            "request": {"id": "long"},
            "problem": {"enunciado": "Prueba\x00control", "solucion": "5"},
            "raw_response": text,
            "error": None,
            "parse_error": None,
        }
    ]
    synthetic_run(tmp_path / "run", rows=rows)
    book = load_workbook(export_run(tmp_path / "run"))
    assert result_map(book)["Enunciado"] == "Prueba\\u0000control"
    assert "Textos largos" in book.sheetnames
    parts = [
        row[3].value
        for row in book["Textos largos"].iter_rows(min_row=2)
        if row[1].value == "Respuesta original"
    ]
    assert "".join(parts) == text
    book.close()


def test_new_run_paths_are_model_and_local_datetime_and_never_overwritten(tmp_path, monkeypatch):
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 26, 19, 21, 34, tzinfo=timezone.utc)

    monkeypatch.setattr("common.files.datetime", Fixed)
    config = {"model": {"id": "HuggingFaceTB/SmolLM3-3B", "path": "missing"}}
    first = new_run(tmp_path, "base_model", config, [], details=True)
    second = new_run(tmp_path, "skill", config, [], details=True)
    assert first == tmp_path / "HuggingFaceTB/SmolLM3-3B/2026-10-26_16-21-34"
    assert second.name == "2026-10-26_16-21-34-02"
    assert (first / "detalles/manifest.json").is_file()
    assert not (first / "manifest.json").exists()
    assert read_json(first / "detalles/manifest.json")["kind"] == "base_model"
    with pytest.raises(ValueError, match="modelID"):
        new_run(tmp_path, "base_model", {"model": {"id": "../escape"}}, [])


def test_cli_can_export_existing_smoke_without_importing_ml(tmp_path):
    import sys

    run = tmp_path / "old-smoke"
    synthetic_run(run)
    result = CliRunner().invoke(app, ["export-excel", "--run", str(run)])
    assert result.exit_code == 0, result.output
    assert str(run / "resultados.xlsx") in result.output
    assert "vllm" not in sys.modules and "torch" not in sys.modules


def test_generation_creates_excel_and_keeps_raw_files_under_details(tmp_path, monkeypatch):
    from common.contracts import Experiment
    from common.generation import infer

    closed = []

    class FakeBench:
        def __init__(self, cfg, run, **kwargs):
            self.run = run

        def metadata(self):
            return {"vllm_version": "synthetic"}

        def render(self, messages):
            return {"input_tokens": 10, "rendered_prompt": "synthetic"}

        def tokenize(self, messages):
            return 10

        def start(self):
            (self.run / "serve.log").write_text("synthetic process")

        def batch(self, messages, **kwargs):
            native = self.run / "native.json"
            write_json(
                native, {"output_throughput": 10.0, "duration": 2.0, "completed": 1, "failed": 0}
            )
            return [
                {
                    "error": None,
                    "raw_response": '{"enunciado":"Prueba sintética", "solucion":"5"}',
                    "output_limit_reached": False,
                    "bench_result": str(native),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }
            ]

        def close(self):
            closed.append(True)

    monkeypatch.setattr("common.runtime.validate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("common.generation.LocalBench", FakeBench)
    cfg = Experiment(
        model={"id": "HuggingFaceTB/SmolLM3-3B", "path": "missing", "revision": "a" * 40},
        benchmark=str(tmp_path / "absent.jsonl"),
        phase="development",
        development="examples/smoke/requests.jsonl",
        request_limit=1,
        repetitions=1,
        output=str(tmp_path / "output"),
    )
    run = infer(cfg, "base_model", 0)
    assert closed == [True]
    assert sorted(p.name for p in run.iterdir()) == ["detalles", "resultados.xlsx"]
    assert read_json(run / "detalles/status.json") == {"status": "complete"}
    book = load_workbook(run / "resultados.xlsx")
    assert result_map(book)["Enunciado"] == "Prueba sintética"
    assert result_map(book)["Tokens salida"] == 20
    book.close()
