#!/usr/bin/env bash
#SBATCH --job-name=tesis-judge-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
revision=${1:?Indica el commit de Gemma}
[[ $revision =~ ^[a-f0-9]{40}$ ]] || { echo "Revisión inválida" >&2; exit 2; }
image=${CONTAINER_IMAGE:-${JUDGE_IMAGE:-docker.io#vllm/vllm-openai:v0.15.1}}
srun --container-image="$image" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$PWD:$PWD,$HOME/.cache/huggingface:$HOME/.cache/huggingface:ro" \
    --container-workdir="$PWD" \
    bash -s -- "$revision" "$HOME/.cache/huggingface/token" <<'BASH'
set -euo pipefail
if ! python3 -c 'import typer, dotenv, psutil, pandas, pydantic'; then
    python3 -m pip install --target "$PWD/.judge-deps" 'typer>=0.27.3,<1' 'python-dotenv>=1,<2' 'pandas>=2,<4'
fi
export PYTHONPATH="$PWD/.judge-deps:${PYTHONPATH:-}"
python3 -c 'import typer, dotenv, psutil, pandas, pydantic'
python3 - "$1" "$2" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

revision, token_path = sys.argv[1:]
path = snapshot_download(
    "google/gemma-3-27b-it", revision=revision, cache_dir="models",
    token=Path(token_path).read_text().strip(),
    allow_patterns=[
        "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "preprocessor_config.json",
        "processor_config.json", "model*.safetensors", "model.safetensors.index.json",
    ],
)
if Path(path).name != revision:
    raise RuntimeError("El snapshot descargado no corresponde al commit solicitado")
print(path)
PY
BASH
