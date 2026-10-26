from pathlib import Path

from common.contracts import Experiment
from common.files import read_json


def load_config(path: Path) -> Experiment:
    """Resolve all data paths relative to the config file's project root."""
    cfg = Experiment.model_validate(read_json(path))
    root = path.resolve().parent.parent
    cfg.model.path = str((root / cfg.model.path).resolve())
    for name in (
        "benchmark",
        "development",
        "oa_registry",
        "skill_review",
        "judge_review",
        "train",
        "validation",
        "demonstrations",
        "skill",
        "output",
        "adapter",
    ):
        value = getattr(cfg, name)
        if value:
            setattr(cfg, name, str((root / value).resolve()))
    if Path(cfg.model.path).is_absolute() and not Path(cfg.model.path).is_dir():
        raise ValueError(f"Modelo local inexistente: {cfg.model.path}")
    return cfg
