#!/bin/bash
#SBATCH --job-name=tesis-judge
#SBATCH --partition=AI
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${JUDGE_IMAGE:-docker.io#vllm/vllm-openai:v0.15.1}}"
CONTAINER_NAME="${CONTAINER_NAME:-tesis-judge-vllm0151}"
cd "$WORKDIR"

if (( $# < 3 )); then
    echo "Uso: sbatch judge.slurm CONFIG JSONL RÚBRICA [opciones del judge]" >&2
    exit 2
fi
for input in "$1" "$2" "$3"; do
    if [[ ! -f "$input" ]]; then
        printf 'Falta el archivo %s\n' "$input" >&2
        exit 2
    fi
done

printf 'Trabajo Slurm %s\nDirectorio %s\nContenedor %s\n' \
    "${SLURM_JOB_ID:-desconocido}" "$WORKDIR" "$CONTAINER_IMAGE"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

srun \
    --container-workdir="$WORKDIR" \
    --container-image="$CONTAINER_IMAGE" \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$WORKDIR:$WORKDIR" \
    --no-container-entrypoint \
    --no-container-mount-home \
    bash -s -- "$@" <<'BASH'
set -euo pipefail
export PYTHONPATH="$PWD/.judge-deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

judge=(python3 main.py judge --config "$1" --source "$2" --rubric "$3")
shift 3
judge+=("$@")
for option in "$@"; do
    if [[ "$option" == --check-only ]]; then
        "${judge[@]}"
        exit
    fi
done

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -c 'from common.runtime import check_inference_runtime; check_inference_runtime()'
"${judge[@]}" --check-only
result=$("${judge[@]}")
printf 'Evaluaciones en %s\n' "$result"
python3 main.py summarize --source "$result/evaluations.jsonl" --output "$result/summary.json"
BASH
