import json

from common.files import append_jsonl, load_rows, sha256
from judge.prepare import prepare


def test_recovery_preserves_raw_rows_and_accepts_missing_fence_only_with_complete_json(tmp_path):
    raw = json.dumps({"enunciado": "P", "descripcion_figura": "", "solucion": "S"})
    group = ["```json\n" + raw + "\n```", "```json\n" + raw, "```json\n" + raw[:-1]]
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    for text in group:
        append_jsonl(source, {"raw_response": text, "parse_error": "Formato", "problem": None})
    assert prepare(source, output) == {"recovered": 2, "format_error": 1}
    rows = load_rows(output)
    assert [row["raw_response"] for row in rows] == group
    assert all(row["parse_error"] == "Formato" for row in rows)
    assert all(row["judge_preparation"]["source_sha256"] == sha256(source) for row in rows)
    assert rows[0]["problem"] == rows[1]["problem"]
    assert rows[2]["problem"] is None
