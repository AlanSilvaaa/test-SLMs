"""Resolve named models into pinned local snapshots before offline execution."""

import re
from pathlib import Path

from common.contracts import Model
from common.files import read_json

DEFAULT_MODELS_DIR = Path("/models")
# Friendly names are deliberately unambiguous; full catalog IDs also work.
ALIASES = {
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "granite-2b": "ibm-granite/granite-3.3-2b-instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "granite-8b": "ibm-granite/granite-3.3-8b-instruct",
}


def model_id(name):
    identifier = ALIASES.get(name.casefold(), name)
    catalog = read_json(Path(__file__).resolve().parent.parent / "configs/models.json")
    if identifier not in catalog["generators"]:
        raise ValueError(f"Modelo desconocido: {name}. Usa SmolLM3 o un ID del catálogo")
    return identifier


def _snapshot_download(**kwargs):
    from common.environment import load_repo_env

    load_repo_env()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Instala dependencias en el equipo experimental: uv sync --extra infer"
        ) from error
    return snapshot_download(**kwargs)


def check_snapshot(path):
    """Older Hub versions may return partially cached snapshots in offline mode."""
    path = Path(path).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise ValueError("La ruta del snapshot no identifica un commit exacto")
    required = [path / "config.json", path / "tokenizer_config.json"]
    if (path / "tokenizer.json").is_file():
        required.append(path / "tokenizer.json")
    else:
        required.append(path / "tokenizer.model")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        required.append(index_path)
        weights = read_json(index_path).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Índice de pesos vacío o inválido")
        for filename in set(weights.values()):
            part = Path(filename)
            if part.is_absolute() or ".." in part.parts:
                raise ValueError("Ruta de shard inválida")
            required.append(path / part)
    else:
        required.append(path / "model.safetensors")
    missing = [p.name for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f"Snapshot incompleto: {', '.join(missing)}")
    read_json(path / "config.json")
    read_json(path / "tokenizer_config.json")
    return path


def prepare_model(name, models_dir=DEFAULT_MODELS_DIR, revision=None, *, offline=False):
    """Prefer the existing snapshot; download missing files only when allowed.

    No weights are loaded here. The caller runs this on the experimental host.
    Subsequent inference always receives a local path and a concrete commit.
    """
    identifier = model_id(name)
    cache = Path(models_dir).expanduser().resolve()
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision requiere un commit exacto de 40 caracteres")
    arguments = {"repo_id": identifier, "cache_dir": str(cache), "revision": revision or "main"}
    try:
        snapshot = check_snapshot(_snapshot_download(**arguments, local_files_only=True))
    except (OSError, ValueError) as error:
        if offline:
            raise ValueError(
                f"No hay un snapshot completo de {identifier} en {cache}. "
                "Prepara los pesos con conexión antes de usar --offline"
            ) from error
        cache.mkdir(parents=True, exist_ok=True)
        # The first setup may use the network. Never change the user's offline
        # environment to force a connection; Hub will honor it if explicitly set.
        snapshot = check_snapshot(_snapshot_download(**arguments, local_files_only=False))
    return Model(id=identifier, path=str(snapshot), revision=snapshot.name)
