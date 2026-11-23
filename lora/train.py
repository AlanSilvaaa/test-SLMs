"""LoRA SFT with explicit response-only masks and fixed, external splits."""

import json
import os
from collections import Counter
from pathlib import Path

from common.contracts import Example, Request
from common.data import validate_development, validate_requests, validate_separation
from common.files import load_rows, new_run, read_json, write_json
from common.prompts import SYSTEM, user_prompt


def encode_example(tokenizer, row, max_length, template_kwargs):
    prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(row)}]
    answer = json.dumps(
        {"enunciado": row["enunciado"], "solucion": row["solucion"]}, ensure_ascii=False
    )
    prefix = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, **template_kwargs
    )
    full = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError(
            "La plantilla no permite aislar la respuesta con una máscara de prefijo. Revisa el tokenizador antes de entrenar"
        )
    if len(full) > max_length:
        raise ValueError(f"{row['id']}: {len(full)} tokens exceden {max_length}. No se truncará")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prefix) + full[len(prefix) :],
    }


class ResponseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        batch = self.tokenizer.pad(
            [{k: f[k] for k in ("input_ids", "attention_mask")} for f in features],
            padding=True,
            return_tensors="pt",
        )
        size = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [f["labels"] + [-100] * (size - len(f["labels"])) for f in features]
        )
        return batch


def train(cfg):
    from common.runtime import validate_model

    validate_model(cfg, judge=False)
    if cfg.seed != 42 or cfg.learning_rate not in {0.00005, 0.0001, 0.0002}:
        raise ValueError("LoRA requiere semilla 42 y una de las tres learning rates")
    if cfg.target_modules != ["all-linear"]:
        raise ValueError("El protocolo requiere atención y MLP completos: all-linear")
    if not cfg.container_image.strip():
        raise ValueError("Registra la identidad del contenedor de entrenamiento")
    # Lazy imports keep --help, data preparation and tests independent of CUDA.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    train_rows = load_rows(cfg.train, Example)
    validation_rows = load_rows(cfg.validation, Example)
    requests = load_rows(cfg.benchmark, Request)
    validate_requests(requests, cfg.oa_registry, legacy=True)
    development = load_rows(cfg.development, Request)
    validate_development(development, requests)
    validate_separation(train_rows, validation_rows, requests)
    validate_separation(train_rows, validation_rows, development)
    train_counts = Counter(r["codigo_oa"] for r in train_rows)
    validation_counts = Counter(r["codigo_oa"] for r in validation_rows)
    if len(train_counts) != 59 or set(train_counts) != set(validation_counts):
        raise ValueError("Se requieren los mismos 59 OA en ambas particiones")
    if set(train_counts.values()) != {34} or set(validation_counts.values()) != {6}:
        raise ValueError("La partición debe contener 34/6 ejemplos por OA")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("El entrenamiento requiere GPU con BF16")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("El protocolo requiere un único proceso y una GPU")
    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path, revision=cfg.model.revision, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [
        [encode_example(tokenizer, r, cfg.max_model_len, cfg.chat_template_kwargs) for r in rows]
        for rows in (train_rows, validation_rows)
    ]
    run = new_run(
        cfg.output,
        "lora-train",
        cfg.model_dump(),
        [
            Path(cfg.train),
            Path(cfg.validation),
            Path(cfg.benchmark),
            Path(cfg.development),
            Path(cfg.oa_registry),
        ],
    )
    write_json(
        run / "mask_audit.json",
        {
            "prompt_tokens_masked": True,
            "padding_label": -100,
            "max_sequence_tokens": max(len(e["input_ids"]) for rows in encoded for e in rows),
            "train_ids": [[r["id"], r["version"]] for r in train_rows],
            "validation_ids": [[r["id"], r["version"]] for r in validation_rows],
        },
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        revision=cfg.model.revision,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    targets = "all-linear" if cfg.target_modules == ["all-linear"] else cfg.target_modules
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
        ),
    )
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError("Se detectaron parámetros entrenables fuera de LoRA")
    write_json(
        run / "hardware.json",
        {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "slurm": {
                k: os.environ.get(k)
                for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME")
            },
            "trainable_parameters": sum(n for _, n in trainable),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "adapted_parameters": [n for n, _ in trainable],
        },
    )
    args = TrainingArguments(
        output_dir=str(run / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=cfg.learning_rate,
        num_train_epochs=5,
        bf16=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0,
        max_grad_norm=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_drop_last=False,
        report_to="none",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(encoded[0]),
        eval_dataset=Dataset.from_list(encoded[1]),
        processing_class=tokenizer,
        data_collator=ResponseCollator(tokenizer),
    )
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            trainer.train()
        finally:
            write_json(
                run / "warnings.json",
                [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
            )
    adapter = run / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(
        adapter / "provenance.json",
        {"base_revision": cfg.model.revision, "base_path": cfg.model.path, "base_id": cfg.model.id},
    )
    trainer.save_state()
    best = min(
        (r for r in trainer.state.log_history if "eval_loss" in r),
        key=lambda r: (r["eval_loss"], r["epoch"]),
    )
    write_json(
        run / "selection.json",
        {
            "model": cfg.model.model_dump(),
            "learning_rate": cfg.learning_rate,
            "eval_loss": best["eval_loss"],
            "epoch": best["epoch"],
            "checkpoint": trainer.state.best_model_checkpoint,
            "adapter": str(adapter.resolve()),
        },
    )
    return run


def select(runs, output):
    """Select only among the three rates of one model and identical data/config."""
    from common.files import sha256

    records = [read_json(run / "selection.json") for run in runs]
    manifests = [read_json(run / "manifest.json") for run in runs]
    rates = {r["learning_rate"] for r in records}
    if len(records) != 3 or rates != {0.00005, 0.0001, 0.0002}:
        raise ValueError("Se requieren las tres learning rates del protocolo")
    first = {
        k: v for k, v in manifests[0]["config"].items() if k not in ("learning_rate", "output")
    }
    for manifest in manifests[1:]:
        other = {
            k: v for k, v in manifest["config"].items() if k not in ("learning_rate", "output")
        }
        if other != first or manifest["inputs"] != manifests[0]["inputs"]:
            raise ValueError("Las ejecuciones no comparten modelo, partición y configuración")
    best = min(records, key=lambda r: (r["eval_loss"], r["epoch"], r["learning_rate"]))
    best["adapter_sha256"] = sha256(Path(best["adapter"]) / "adapter_model.safetensors")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2)
