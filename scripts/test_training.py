"""Offline integration: construct a tiny random Qwen2, train LoRA, reload export.

This validates plumbing, not medical performance. No network or model weights needed.
"""

import json
import tempfile
from pathlib import Path


def main():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from healthcare_lora.common import LABEL_NAMES, write_json, write_jsonl
    from healthcare_lora.train import train
    from healthcare_lora.model import HFPair

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model_path = root / "tiny-qwen"
        tokens = (
            ["<pad>", "<eos>", "<unk>", "<start>", "<end>"]
            + LABEL_NAMES
            + [
                "system",
                "user",
                "assistant",
                "Sentence",
                "Classify",
                "the",
                "medical",
                "research",
                "patients",
                "randomized",
                "outcomes",
                "results",
                "study",
                ":",
                ".",
                "groups",
            ]
        )
        vocab = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
        tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
        )
        tokenizer.chat_template = "{% for message in messages %}{{ '<start>' + message['role'] + ': ' + message['content'] + '<end>' }}{% endfor %}{% if add_generation_prompt %}{{ '<start>assistant: ' }}{% endif %}"
        tokenizer.save_pretrained(model_path)
        config = Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=vocab["<start>"],
            eos_token_id=vocab["<eos>"],
            pad_token_id=vocab["<pad>"],
        )
        Qwen2ForCausalLM(config).save_pretrained(model_path)
        data = root / "data"
        for split in ["train", "validation", "test"]:
            write_jsonl(
                data / f"{split}.jsonl",
                [
                    {
                        "id": f"{split}-{i}",
                        "group_id": f"{split}-{i}",
                        "text": "The patients were randomized into study groups.",
                        "label": label,
                    }
                    for i, label in enumerate(LABEL_NAMES)
                ],
            )
        cfg = {
            "model": str(model_path),
            "revision": None,
            "rank": 2,
            "alpha": 4,
            "dropout": 0,
            "learning_rate": 0.001,
            "epochs": 1,
            "max_length": 256,
            "seed": 42,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "gradient_checkpointing": False,
            "device": "cpu",
            "qlora": False,
            "target_modules": ["q_proj", "v_proj"],
            "output": str(root / "run"),
            "logging_steps": 1,
            "tracking_uri": f"sqlite:///{root}/mlflow.db",
        }
        cfg_path = root / "config.json"
        write_json(cfg_path, cfg)
        manifest = train(cfg_path, data=str(data))
        assert manifest["parameter_counts"]["trainable"] < manifest["parameter_counts"]["total"]
        assert manifest["best_checkpoint"]
        assert (root / "run" / "adapter" / "adapter_model.safetensors").exists()
        pair = HFPair(cfg, str(root / "run" / "adapter"))
        pair.predict("The patients were randomized into study groups.", adapted=True)
        pair.predict("The patients were randomized into study groups.", adapted=False)
        result = {
            "status": "passed",
            "scope": "tiny random local model; no domain-performance claim",
            "trainable_parameters": manifest["parameter_counts"]["trainable"],
            "total_parameters": manifest["parameter_counts"]["total"],
            "training_seconds": manifest["duration_seconds"],
            "packages": manifest["packages"],
        }
        write_json("results/training_check.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
