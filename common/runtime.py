"""Validate model identity and collect host metadata without loading models."""

import re
import subprocess
from pathlib import Path

import psutil

from common.files import read_json


def validate_torch_runtime(torch):
    if not torch.version.cuda:
        raise RuntimeError(
            f"PyTorch {torch.__version__} es una instalación solo CPU. "
            "En GX10/ARM64 se requiere el entorno CUDA 13 configurado por el proyecto. "
            "Actualiza el repositorio y ejecuta uv sync --locked --extra infer"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} incluye CUDA {torch.version.cuda}, "
            "pero no puede acceder a la GPU. Comprueba el controlador y el acceso a la GPU"
        )


def check_inference_runtime():
    # Device discovery only; no model loading, inference or GPU computation.
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "No se pudo importar PyTorch. Prepara las dependencias CUDA del equipo experimental: "
            "uv sync --locked --extra infer"
        ) from error
    validate_torch_runtime(torch)


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
