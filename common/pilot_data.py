"""Read the thesis pilot workbooks without accepting candidates or finalizing data."""

import re
import shutil
from pathlib import Path

from common.contracts import Request
from common.data import AXES, normalized, validate_development, validate_requests
from common.files import append_jsonl, load_rows, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / "data/pilot"


def sheet_rows(book, sheet):
    rows = iter(book[sheet].values)
    header = next(rows)
    return [dict(zip(header, row, strict=True)) for row in rows if row and row[0]]


def course(value):
    match = re.fullmatch(r"([1-6])° básico", str(value))
    if not match:
        raise ValueError(f"Curso no reconocido: {value}")
    return match[1]


def prepare_pilot(source, oa_source, output=DEFAULT_PILOT):
    from openpyxl import load_workbook

    source, oa_source, output = Path(source), Path(oa_source), Path(output)
    if output.exists():
        raise FileExistsError(
            f"Ya existe {output}; elige otra carpeta para conservar la versión anterior"
        )
    with source.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            candidates = sheet_rows(book, "Dataset piloto")
            benchmark = sheet_rows(book, "Benchmark piloto")
        finally:
            book.close()
    pilot_codes = {row["codigo_oa"] for row in benchmark}
    with oa_source.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            objectives = [
                {
                    "codigo_oa": row["codigo_oa"],
                    "curso": course(row["curso"]),
                    "eje": AXES[row["eje"]],
                    "texto_oa": row["texto_original"],
                }
                for row in sheet_rows(book, "OA")
                if row["codigo_oa"] in pilot_codes
            ]
        finally:
            book.close()
    by_code = {oa["codigo_oa"]: oa for oa in objectives}

    def convert(rows, id_column):
        converted = []
        for row in rows:
            oa = by_code[row["codigo_oa"]]
            if course(row["curso"]) != oa["curso"]:
                raise ValueError(f"{row[id_column]}: curso distinto del registro curricular")
            if normalized(oa["texto_oa"]) not in normalized(row["solicitud"]):
                raise ValueError(f"{row[id_column]}: texto OA distinto del registro curricular")
            fields = {
                key: row[key]
                for key in (
                    "version",
                    "codigo_oa",
                    "habilidad_evaluada",
                    "condiciones_curriculares",
                    "contexto",
                    "solicitud",
                )
            }
            fields.update(id=row[id_column], curso=oa["curso"], texto_oa=oa["texto_oa"])
            converted.append(Request.model_validate(fields).model_dump())
        if len({r["id"] for r in converted}) != len(converted):
            raise ValueError("Identificadores duplicados en el piloto")
        return converted

    requests = convert(benchmark, "id_solicitud")
    exclusions = convert(candidates, "id_problema")
    if len(exclusions) != 59 or {r["codigo_oa"] for r in exclusions} != set(by_code):
        raise ValueError("El dataset piloto debe representar los 59 OA")
    validate_development(exclusions, requests)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the destination exclusively; remove only our own partial preparation on failure.
    output.mkdir(exist_ok=False)
    try:
        write_json(
            output / "oa.json", {"version": "pilot-v1", "status": "draft", "objectives": objectives}
        )
        validate_requests(requests, output / "oa.json", legacy=True, require_approved=False)
        for name, rows in (("benchmark.jsonl", requests), ("dataset-requests.jsonl", exclusions)):
            for row in rows:
                append_jsonl(output / name, row)
        write_json(
            output / "manifest.json",
            {
                "format": "test-slms-pilot-v1",
                "status": "draft",
                "review": "Pendiente de revisión curricular y de similitudes; no es evaluación final",
                "sources": {source.name: sha256(source), oa_source.name: sha256(oa_source)},
                "sheets": ["Benchmark piloto", "Dataset piloto", "OA"],
                "counts": {"objectives": 59, "requests": 295, "requests_per_oa": 5},
                "files": {
                    name: sha256(output / name)
                    for name in ("oa.json", "benchmark.jsonl", "dataset-requests.jsonl")
                },
            },
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def validate_pilot(cfg):
    if not cfg.pilot_manifest:
        raise ValueError("Falta el manifiesto del paquete piloto")
    path = Path(cfg.pilot_manifest)
    manifest = read_json(path)
    names = {"oa.json", "benchmark.jsonl", "dataset-requests.jsonl"}
    if (
        manifest.get("format") != "test-slms-pilot-v1"
        or manifest.get("status") != "draft"
        or set(manifest.get("files", {})) != names
    ):
        raise ValueError("Paquete piloto desconocido; vuelve a prepararlo desde los Excel")
    for name in names:
        if sha256(path.parent / name) != manifest["files"][name]:
            raise ValueError(f"El paquete piloto cambió: {name}")
    if (
        Path(cfg.benchmark).resolve() != (path.parent / "benchmark.jsonl").resolve()
        or Path(cfg.oa_registry).resolve() != (path.parent / "oa.json").resolve()
    ):
        raise ValueError("Las rutas no corresponden al paquete piloto")
    requests = load_rows(cfg.benchmark, Request)
    validate_requests(requests, cfg.oa_registry, legacy=True, require_approved=False)
    validate_development(load_rows(path.parent / "dataset-requests.jsonl", Request), requests)
    validate_development(load_rows(cfg.development, Request), requests)
    return [path, *(path.parent / name for name in sorted(names))]
