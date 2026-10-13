import contextlib
import json
import time
from pathlib import Path
from .common import messages, file_hash, PROMPT_VERSION
from .metrics import parse_label


def render_input(tokenizer, text, max_length):
    """Keep the chat delimiters intact; shrink only the abstract when it exceeds the budget."""
    original = text
    budget = max_length - 32
    if budget < 64:
        raise ValueError("max_length must be at least 96")
    for _ in range(30):
        rendered = tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if count <= budget:
            return rendered, text != original
        text = text[: max(0, int(len(text) * (budget / count) * 0.95))]
    raise ValueError("Prompt instruction alone exceeds input budget")


def resolve_device(choice="auto"):
    import torch

    if choice != "auto":
        if choice == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if choice == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        return choice
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class HFPair:
    """One base model with an optional adapter. Serialize calls at the API boundary."""

    evidence_kind = "real_model"

    def __init__(self, config, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.adapter = adapter
        self.device = resolve_device(config.get("device", "auto"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config.get("revision"), trust_remote_code=False
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config.get("revision"),
            trust_remote_code=False,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        if adapter:
            from peft import PeftModel

            manifest_path = Path(adapter) / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("Adapter manifest is missing; use a Healthcare LoRA training export")
            manifest = json.loads(manifest_path.read_text())
            if manifest["model"] != config["model"] or manifest.get("revision") != config.get("revision"):
                raise ValueError("Adapter base model identity does not match config")
            if manifest.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("Adapter task prompt version does not match inference")
            if manifest["max_length"] != config["max_length"]:
                raise ValueError("Inference max_length differs from adapter training")
            for name, expected in manifest["adapter_sha256"].items():
                if file_hash(Path(adapter) / name) != expected:
                    raise ValueError(f"Adapter checksum mismatch: {name}")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, chat, adapted=False, is_rendered=False, max_new_tokens=24):
        import torch

        if adapted and not self.adapter:
            raise ValueError("Adapted generation requires a trained adapter")
        rendered = (
            chat
            if is_rendered
            else self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        if inputs.input_ids.shape[1] > self.config["max_length"] - max_new_tokens:
            raise ValueError("Input exceeds configured context budget")
        inputs.pop("token_type_ids", None)
        context = self.model.disable_adapter() if self.adapter and not adapted else contextlib.nullcontext()
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with context, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - start) * 1000
        generated = output[0, inputs.input_ids.shape[1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "raw_output": raw,
            "latency_ms": ms,
            "input_tokens": int(inputs.input_ids.shape[1]),
            "output_tokens": len(generated),
            "tokens_per_second": len(generated) / (ms / 1000) if ms else 0,
            "confidence": None,
        }

    def predict(self, text, adapted=False):
        rendered, truncated = render_input(self.tokenizer, text, self.config["max_length"])
        result = self.generate(rendered, adapted=adapted, is_rendered=True)
        return {**result, "prediction": parse_label(result["raw_output"]), "input_truncated": truncated}


class HFView:
    evidence_kind = "real_model"

    def __init__(self, pair, adapted=False):
        self.pair, self.adapted = pair, adapted
        self.name = "lora-adapter" if adapted else "base-llm"

    def predict(self, text):
        return self.pair.predict(text, self.adapted)
