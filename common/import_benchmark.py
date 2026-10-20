"""Explicit adapter for the original MathGenBench-CL request format."""

from pathlib import Path

from common.contracts import Request
from common.files import append_jsonl, load_rows


def convert(source: Path, output: Path):
    rows = load_rows(source)
    converted = []
    for row in rows:
        fields = {
            key: row[key]
            for key in (
                "id",
                "curso",
                "codigo_oa",
                "texto_oa",
                "habilidad_evaluada",
                "condiciones_curriculares",
                "contexto",
            )
        }
        fields["version"] = row.get("version", 1)
        fields["solicitud"] = row.get("solicitud") or (
            f"Crea un problema matemático verbal para {row['curso']} que trabaje "
            f"{row['habilidad_evaluada']} del OA {row['texto_oa']}. "
            f"Respeta {row['condiciones_curriculares']} y sitúalo en {row['contexto']}. "
            "El enunciado debe tener una respuesta única y verificable. "
            "Devuelve enunciado y solucion. En solucion explica el procedimiento y entrega la respuesta final."
        )
        converted.append(Request.model_validate(fields).model_dump())
    # Validate the entire source before creating the destination.
    keys = [(r["id"], r["version"]) for r in converted]
    if len(keys) != len(set(keys)):
        raise ValueError("IDs y versiones duplicados")
    with output.open("x", encoding="utf-8"):
        pass
    for row in converted:
        append_jsonl(output, row)
