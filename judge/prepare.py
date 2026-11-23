"""Recover machine-readable output without changing generated content or row order."""

import json
import re
from collections import Counter
from pathlib import Path

from common.contracts import GeneratedProblem
from common.files import load_rows, sha256

OPEN_FENCE = re.compile(r"^```(?:json)?[^\S\r\n]*\r?\n", re.IGNORECASE)
FIELDS = {"enunciado", "descripcion_figura", "solucion"}


def recover(raw):
    text = raw.strip()
    fence = OPEN_FENCE.match(text)
    if fence:
        text = text[fence.end() :].strip()
        if text.endswith("\n```"):
            text = text[:-3].rstrip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Se requieren enunciado, descripcion_figura y solucion")
    if any(not isinstance(value[key], str) for key in FIELDS):
        raise ValueError("Los tres campos deben ser cadenas de texto")
    return GeneratedProblem.model_validate(value).model_dump()


def prepare(source: Path, output: Path):
    rows = load_rows(source)
    counts = Counter()
    source_hash = sha256(source)
    for index, row in enumerate(rows):
        if row.get("problem") is not None:
            outcome = "original"
        elif row.get("error"):
            outcome = "execution_error"
        else:
            try:
                row["problem"] = recover(row.get("raw_response") or "")
                outcome = "recovered"
            except (ValueError, TypeError, AttributeError):
                outcome = "format_error"
        row["judge_preparation"] = {
            "outcome": outcome,
            "source_index": index,
            "source_sha256": source_hash,
        }
        counts[outcome] += 1
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)
