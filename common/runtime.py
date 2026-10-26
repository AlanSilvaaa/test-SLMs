"""Validate model identity and collect host metadata without loading models."""

import re
import subprocess
from pathlib import Path

import psutil

from common.files import read_json


def validate_model(cfg, *, judge):
    catalog = read_json(Path(__file__).resolve().parent.parent / "configs/models.json")
    allowed = [catalog["judge"]] if judge else catalog["generators"]
    if cfg.model.id not in allowed:
        raise ValueError("model.id debe corresponder a un modelo de la memoria")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.model.revision):
        raise ValueError("model.revision requiere el commit exacto de 40 caracteres")
    if not Path(cfg.model.path).is_absolute() or not Path(cfg.model.path).is_dir():
        raise ValueError("model.path debe ser una carpeta absoluta local")


def hardware_metadata():
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        gpu_info = {"returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_info = {"error": str(error)}
    return {
        "gpu": gpu_info,
        "host_memory_bytes": psutil.virtual_memory().total,
        "cpu_count": psutil.cpu_count(),
        "load_average": psutil.getloadavg(),
    }
