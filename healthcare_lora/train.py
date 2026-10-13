"""Completion-only LoRA SFT with early stopping, tracking, and adapter provenance."""

import importlib.metadata
import json
import random
import time
from pathlib import Path
from .common import environment, file_hash, read_jsonl, write_json
from .data import assert_disjoint, stratified_take
from .model import render_input, resolve_device


def train(config_path, data="data/processed", output=None):
    import mlflow
    import torch
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    cfg = json.loads(Path(config_path).read_text())
    required = [
        "model",
        "revision",
        "rank",
        "alpha",
        "dropout",
        "learning_rate",
        "epochs",
        "max_length",
        "seed",
    ]
    if any(k not in cfg for k in required):
        raise ValueError("Training config is incomplete")
    if cfg["rank"] <= 0 or cfg["alpha"] <= 0 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Invalid LoRA hyperparameters")
    root = Path(output or cfg["output"])
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    device = resolve_device(cfg.get("device", "auto"))
    if cfg.get("qlora", False) and device != "cuda":
        raise ValueError("The provided QLoRA path requires an NVIDIA CUDA device")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = Path(data)
    splits = {k: read_jsonl(data / f"{k}.jsonl") for k in ["train", "validation"]}
    # Detect accidental test leakage using group hashes without selecting on test labels.
    splits["test"] = read_jsonl(data / "test.jsonl")
    assert_disjoint(splits)
    rows = {k: stratified_take(splits[k], cfg.get(f"{k}_limit"), seed) for k in ["train", "validation"]}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    prepared, truncation = {}, {}
    for split, items in rows.items():
        formatted, trimmed = [], 0
        for row in items:
            rendered, was_trimmed = render_input(tokenizer, row["text"], cfg["max_length"])
            trimmed += was_trimmed
            formatted.append({"prompt": rendered, "completion": row["label"] + tokenizer.eos_token})
        prepared[split] = Dataset.from_list(formatted)
        truncation[split] = trimmed
    kwargs = {
        "revision": cfg["revision"],
        "trust_remote_code": False,
        "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
    }
    if cfg.get("qlora", False):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kwargs)
    if cfg.get("qlora", False):
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)
    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
    )
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Healthcare-LoRA")
    history = []

    class AuditCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if device == "cuda":
                values["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            elif device == "mps":
                values["mps_allocated_memory_mb"] = torch.mps.current_allocated_memory() / 1024**2
            mlflow.log_metrics(values, step=state.global_step)
            history.append({"step": state.global_step, **values})

    args = SFTConfig(
        output_dir=str(root / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation", 4),
        max_length=cfg["max_length"],
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
        logging_steps=cfg.get("logging_steps", 5),
        report_to="none",
        fp16=device == "cuda",
        bf16=False,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda",
        max_steps=cfg.get("max_steps", -1),
    )
    with mlflow.start_run(run_name=root.name) as run:
        mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in cfg.items()})
        write_json(root / "config.json", cfg)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=prepared["train"],
            eval_dataset=prepared["validation"],
            processing_class=tokenizer,
            peft_config=lora,
            callbacks=[AuditCallback(), EarlyStoppingCallback(early_stopping_patience=2)],
        )
        counts = {
            "trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in trainer.model.parameters()),
        }
        mlflow.log_metrics(counts)
        start = time.perf_counter()
        trainer.train()
        duration = time.perf_counter() - start
        validation = trainer.evaluate()
        adapter = root / "adapter"
        trainer.model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        from .common import PROMPT_VERSION

        manifest = {
            "evidence_kind": "real_lora_training",
            "model": cfg["model"],
            "revision": cfg["revision"],
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "prompt_version": PROMPT_VERSION,
            "max_length": cfg["max_length"],
            "config": cfg,
            "environment": environment(),
            "device": device,
            "mlflow_run_id": run.info.run_id,
            "duration_seconds": duration,
            "parameter_counts": counts,
            "validation": validation,
            "selection_metric": "validation_loss",
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "truncation_counts": truncation,
            "source_hashes": {k: file_hash(data / f"{k}.jsonl") for k in ["train", "validation"]},
            "selected_example_ids": {k: [r["id"] for r in v] for k, v in rows.items()},
            "packages": {
                k: importlib.metadata.version(k) for k in ["torch", "transformers", "peft", "trl", "datasets"]
            },
            "adapter_sha256": {p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()},
        }
        write_json(adapter / "manifest.json", manifest)
        write_json(root / "history.json", history)
        mlflow.log_metric("duration_seconds", duration)
        mlflow.log_artifacts(str(root))
    return manifest
