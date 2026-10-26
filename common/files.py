"""UTF-8 artifacts with checksums and non-overwriting run directories."""

import hashlib
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import ZoneInfo


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path, contract=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: se requiere una lista no vacía")
    if contract:
        rows = [contract.model_validate(row).model_dump() for row in rows]
        keys = [(row["id"], row["version"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path}: identificadores y versiones duplicados")
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run(root, label, config, inputs, *, details=False):
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("America/Santiago"))
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    model = config.get("model") or {}
    namespace = model.get("id") or Path(model.get("path", label)).name
    parts = namespace.split("/")
    if any(p in ("", ".", "..") or not re.fullmatch(r"[\w.-]+", p) for p in parts):
        raise ValueError("modelID inválido para la carpeta de resultados")
    parent = Path(root).joinpath(*parts)
    parent.mkdir(parents=True, exist_ok=True)
    readable = local.strftime("%Y-%m-%d_%H-%M-%S")
    for number in range(1, 1001):
        run = parent / (readable if number == 1 else f"{readable}-{number:02d}")
        try:
            run.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("No se pudo reservar una carpeta nueva para la ejecución")
    artifacts = run / "detalles" if details else run
    artifacts.mkdir(exist_ok=True)
    packages = {}
    for name in ("vllm", "torch", "transformers", "peft", "trl", "pydantic", "pandas", "openpyxl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    model = config.get("model")
    if isinstance(model, dict) and model.get("path") and Path(model["path"]).is_dir():
        # Preserve a fingerprint of the actual local files, including the weights.
        inputs = list(inputs) + [
            p
            for p in sorted(Path(model["path"]).rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        ]
    root = Path(__file__).resolve().parent.parent
    sources = [root / "main.py", root / "pyproject.toml", root / "uv.lock"]
    for directory in ("common", "base_model", "few_shot", "skill", "lora", "judge"):
        sources.extend((root / directory).glob("*.py"))
    for source in sorted(sources):
        if source.is_file():
            target = artifacts / "source" / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_json(
        artifacts / "manifest.json",
        {
            "created_utc": stamp,
            "created_local": local.isoformat(),
            "timezone": "America/Santiago",
            "kind": label,
            "config": config,
            "trl": "not_used; Transformers Trainer with explicit response masks",
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "inputs": {str(p): sha256(p) for p in inputs},
            "source_sha256": {
                str(p.relative_to(root)): sha256(p) for p in sorted(sources) if p.is_file()
            },
        },
    )
    from common.runtime import hardware_metadata

    write_json(artifacts / "hardware-host.json", hardware_metadata())
    return run
