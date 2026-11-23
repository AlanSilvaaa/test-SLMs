"""Portable Excel export for the experimental host; no model/GPU dependencies."""

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from common.files import load_rows, read_json

RESULT_COLUMNS = [
    ("Número", "number", 10),
    ("Estado técnico", "state", 28),
    ("Curso", "curso", 10),
    ("Código OA", "codigo_oa", 20),
    ("Enunciado", "enunciado", 65),
    ("Solución", "solucion", 65),
    ("Contexto solicitado", "contexto", 30),
    ("Estrategia", "module", 18),
    ("Ejemplos few-shot", "shots", 14),
    ("Repetición (desde 1)", "repetition", 18),
    ("Fase", "phase", 16),
    ("ID solicitud", "request_id", 24),
    ("Versión solicitud", "request_version", 14),
    ("Solicitud", "solicitud", 65),
    ("Habilidad evaluada", "habilidad_evaluada", 45),
    ("Condiciones curriculares", "condiciones_curriculares", 50),
    ("Texto OA", "texto_oa", 60),
    ("Tokens entrada", "prompt_tokens", 16),
    ("Tokens salida", "completion_tokens", 16),
    ("Tokens adicionales", "additional_input_tokens", 18),
    ("Latencia total (s)", "latency_s", 18),
    ("Primer token TTFT (s)", "ttft_s", 20),
    ("Decode (s)", "decode_s", 16),
    ("TPOT (s/token)", "tpot_s", 18),
    ("Velocidad decode (tokens/s)", "decode_tokens_per_second", 23),
    ("Límite de salida alcanzado", "output_limit_reached", 22),
    ("Error de ejecución", "error", 50),
    ("Error de formato", "parse_error", 50),
    ("Respuesta original", "raw_response", 65),
    ("Modelo", "model_id", 40),
    ("Revisión del modelo", "model_revision", 44),
    ("Tipo de representación", "tipo_representacion", 28),
    ("Descripción de figura", "descripcion_figura", 65),
]
METRICS = {
    "ttft_s": ("Tiempo hasta el primer token", "s"),
    "latency_s": ("Latencia total", "s"),
    "decode_s": ("Tiempo de decode", "s"),
    "tpot_s": ("Tiempo por token de salida", "s/token"),
    "decode_tokens_per_second": ("Velocidad de decode", "tokens/s"),
}


def run_artifacts(run):
    run = Path(run).resolve()
    if run.name == "detalles" and (run / "manifest.json").is_file():
        return run.parent, run
    return run, run / "detalles" if (run / "detalles/manifest.json").is_file() else run


def _optional(folder, name):
    path = folder / name
    return read_json(path) if path.is_file() else {}


def _state(row):
    if row.get("error"):
        return "Error de ejecución"
    if row.get("parse_error") or not row.get("problem"):
        return "Error de formato"
    if row.get("output_limit_reached"):
        return "Límite de salida alcanzado"
    if (row.get("request") or {}).get("tipo_representacion") == "Figura especificada" and not row[
        "problem"
    ].get("descripcion_figura"):
        return "Descripción de figura ausente"
    if not row["problem"].get("solucion"):
        return "Solución ausente"
    return "Sin errores técnicos"


def result_values(row, index, config):
    request, problem = row.get("request") or {}, row.get("problem") or {}
    model = row.get("model") or config.get("model") or {}
    return {
        **request,
        **problem,
        **row,
        **(row.get("usage") or {}),
        "number": index,
        "state": _state(row),
        "request_id": request.get("id"),
        "request_version": request.get("version"),
        "enunciado": problem.get("enunciado"),
        "solucion": problem.get("solucion"),
        "repetition": row["repetition"] + 1 if row.get("repetition") is not None else None,
        "phase": row.get("phase", config.get("phase")),
        "module": row.get("module", config.get("module")),
        "shots": row.get("shots", config.get("shots")),
        "model_id": model.get("id") or model.get("path"),
        "model_revision": model.get("revision"),
    }


def _local_date(manifest):
    value = manifest.get("created_local")
    if value:
        parsed = datetime.fromisoformat(value)
    else:
        parsed = datetime.strptime(manifest["created_utc"], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    return parsed.astimezone(ZoneInfo("America/Santiago")).replace(tzinfo=None)


def export_run(run, output=None, *, status=None):
    """Export new or legacy runs without moving or replacing the source data."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    run, artifacts = run_artifacts(run)
    manifest = read_json(artifacts / "manifest.json")
    config = manifest["config"]
    path = artifacts / "generations.jsonl"
    rows = load_rows(path) if path.is_file() and path.stat().st_size else []
    performance = _optional(artifacts, "performance.json")
    memory = _optional(artifacts, "memory.json")
    engine = _optional(artifacts, "engine.json")
    hardware = _optional(artifacts, "hardware-host.json")
    status = status or _optional(artifacts, "status.json").get(
        "status", "incompleta o interrumpida"
    )
    output = Path(output).resolve() if output is not None else run / "resultados.xlsx"
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; usa --output con otro nombre")

    book = Workbook()
    results = book.active
    results.title = "Resultados"
    summary = book.create_sheet("Resumen")
    navy, pale = "243B53", "EFF4F8"
    body_font = Font(name="Arial", size=11, color="243B53")
    long_text = None
    escaped_controls = False

    def escaped(value):
        # Invalid XML codepoints are represented explicitly, never discarded.
        nonlocal escaped_controls
        safe = []
        for character in value:
            point = ord(character)
            if (
                character in "\t\n\r"
                or 32 <= point <= 0xD7FF
                or 0xE000 <= point <= 0xFFFD
                or point >= 0x10000
            ):
                safe.append(character)
            else:
                safe.append(f"\\u{point:04x}")
                escaped_controls = True
        return "".join(safe)

    def text_value(cell, value):
        # Force model-generated text to a string, including leading '='.
        cell.value = value
        cell.data_type = "s"

    def put(sheet, row, column, value, record=""):
        nonlocal long_text
        cell = sheet.cell(row, column)
        cell.font = body_font
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            value = escaped(value)
        # Conservative UTF-16 bound, also preserving astral Unicode characters.
        if isinstance(value, str) and len(value) > 16000:
            if long_text is None:
                long_text = book.create_sheet("Textos largos")
                long_text.append(["Referencia", "Campo", "Parte", "Texto completo por partes"])
            reference = f"{sheet.title}!{cell.coordinate}"
            for part, offset in enumerate(range(0, len(value), 16000), 1):
                target = long_text.max_row + 1
                long_text.append([reference, record, part])
                text_value(long_text.cell(target, 4), value[offset : offset + 16000])
            value = f"Texto completo en la hoja Textos largos, referencia {reference}"
        if isinstance(value, str):
            text_value(cell, value)
        else:
            cell.value = value
            cell.alignment = Alignment(
                vertical="top", horizontal="right" if value is not None else "left"
            )
            if isinstance(value, datetime):
                cell.number_format = "yyyy-mm-dd hh:mm:ss"
            elif isinstance(value, float):
                cell.number_format = "0.000"
            elif isinstance(value, int) and not isinstance(value, bool):
                cell.number_format = "#,##0"
        return cell

    for column, (label, key, width) in enumerate(RESULT_COLUMNS, 1):
        put(results, 1, column, label)
        results.column_dimensions[get_column_letter(column)].width = width
    for index, row in enumerate(rows, 1):
        values = result_values(row, index, config)
        for column, (label, key, width) in enumerate(RESULT_COLUMNS, 1):
            put(results, index + 1, column, values.get(key), label)
        # Fit typical full problem/solution text; preserve all text for longer cells.
        lines = max(
            len(str(values.get(k) or "")) / 80
            for k in ("enunciado", "solucion", "descripcion_figura")
        )
        results.row_dimensions[index + 1].height = min(360, max(80, 20 * (int(lines) + 2)))
    results.freeze_panes = "E2"
    results.sheet_properties.tabColor = navy
    last_column = get_column_letter(len(RESULT_COLUMNS))
    if rows:
        table = Table(displayName="Problemas", ref=f"A1:{last_column}{len(rows) + 1}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        results.add_table(table)
    else:
        results.auto_filter.ref = f"A1:{last_column}1"

    put(summary, 2, 1, "Resumen de la ejecución").font = Font(
        name="Arial", size=15, bold=True, color=navy
    )
    current = 4

    def entry(label, value, unit=None):
        nonlocal current
        put(summary, current, 1, label)
        put(summary, current, 2, value, label)
        put(summary, current, 3, unit)
        if current % 2 == 0:
            for cell in summary[current]:
                cell.fill = PatternFill("solid", fgColor=pale)
        current += 1

    model = config.get("model") or {}
    entry("Modelo", model.get("id") or model.get("path"))
    entry("Revisión del modelo", model.get("revision"))
    entry("Fecha de ejecución", _local_date(manifest), "America/Santiago")
    entry("Estado de ejecución", "Completa" if status == "complete" else status)
    entry("Estrategia", config.get("module") or manifest.get("kind"))
    entry("Fase", config.get("phase"))
    entry("Registros exportados", len(rows), "generaciones")
    entry("Enunciados estructurados", sum(bool(r.get("problem")) for r in rows), "generaciones")
    entry("Errores de ejecución", sum(bool(r.get("error")) for r in rows), "generaciones")
    entry("Errores de formato", sum(_state(r) == "Error de formato" for r in rows), "generaciones")
    entry(
        "Límite de salida alcanzado",
        sum(bool(r.get("output_limit_reached")) for r in rows),
        "generaciones",
    )
    entry(
        "Soluciones ausentes",
        sum(bool(r.get("problem")) and not r["problem"].get("solucion") for r in rows),
        "generaciones",
    )
    entry("Puntajes pedagógicos", "No calculados en esta exportación")
    entry("Celdas vacías", "Dato no disponible; no equivale a cero")
    for key in (
        "seed",
        "repetitions",
        "shots",
        "temperature",
        "top_p",
        "max_model_len",
        "max_tokens",
        "gpu_memory_utilization",
        "chat_template_kwargs",
    ):
        if key in config:
            entry(f"Configuración: {key}", config[key])
    for key, value in manifest.get("packages", {}).items():
        entry(f"Versión: {key}", value)
    entry("vLLM efectivo", engine.get("vllm_version"))
    entry("Precisión de inferencia", engine.get("dtype"))
    entry("GPU y controlador", (hardware.get("gpu") or {}).get("stdout"))
    for metric, (label, unit) in METRICS.items():
        for statistic, title in (
            ("mean", "media"),
            ("median", "mediana"),
            ("std", "desviación estándar poblacional"),
            ("n", "observaciones"),
        ):
            value = performance.get("per_problem", {}).get(metric, {}).get(statistic)
            entry(f"{label}: {title}", value, "generaciones" if statistic == "n" else unit)
    for statistic, title in (
        ("mean", "media"),
        ("median", "mediana"),
        ("std", "desviación estándar poblacional"),
    ):
        entry(
            f"Tasa agregada de salida: {title}",
            performance.get("aggregate_output_tokens_per_second", {}).get(statistic),
            "tokens/s",
        )
    for repetition, native in performance.get("repetitions", {}).items():
        entry(
            f"Tasa agregada de salida, repetición {int(repetition) + 1}",
            native.get("aggregate_output_tokens_per_second"),
            "tokens/s",
        )
        entry(
            f"Duración benchmark, repetición {int(repetition) + 1}",
            native.get("benchmark_duration_s"),
            "s",
        )
    for phase, title in (("baseline", "inicial"), ("peak", "máxima observada")):
        for key, label in (
            ("host_used_bytes", "RAM del host"),
            ("swap_used_bytes", "Swap"),
            ("process_tree_rss_bytes", "RSS de procesos"),
        ):
            size = memory.get(phase, {}).get(key)
            entry(f"{label}: {title}", size / 1024**3 if size is not None else None, "GiB")
    if memory.get("errors"):
        entry("Errores al medir memoria", memory["errors"])
    if escaped_controls:
        entry(
            "Caracteres de control",
            "Representados como escapes Unicode para conservar su contenido",
        )
    summary.column_dimensions["A"].width = 58
    summary.column_dimensions["B"].width = 72
    summary.column_dimensions["C"].width = 24
    summary.sheet_properties.tabColor = "829AB1"
    for row in range(4, current):
        value = str(summary.cell(row, 2).value or "")
        summary.row_dimensions[row].height = max(24, min(180, 18 * (len(value) // 65 + 1)))

    for sheet in book:
        sheet.sheet_view.showGridLines = False
        if sheet is results or sheet is long_text:
            sheet.row_dimensions[1].height = 40
            for cell in sheet[1]:
                cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor=navy)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
    if long_text is not None:
        long_text.column_dimensions["A"].width = 28
        long_text.column_dimensions["B"].width = 30
        long_text.column_dimensions["C"].width = 10
        long_text.column_dimensions["D"].width = 100
        long_text.freeze_panes = "D2"
        long_text.auto_filter.ref = long_text.dimensions
        for row in long_text.iter_rows(min_row=2):
            for cell in row:
                cell.font = body_font
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            long_text.row_dimensions[row[0].row].height = 200
    book.properties.creator = "test-SLMs"
    book.properties.title = f"Resultados: {model.get('id', '')}"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous exports, including legacy runs.
    with output.open("xb") as handle:
        try:
            book.save(handle)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return output
