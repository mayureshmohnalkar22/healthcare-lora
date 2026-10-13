import json
from pathlib import Path
from .common import read_jsonl, write_json
from .data import stratified_take
from .baseline import predictions
from .metrics import score
from .model import HFPair, HFView
from .train import train


def sweep(configs, data="data/processed", output="results/experiments/sweep.json"):
    results = []
    for config in configs:
        manifest = train(config, data=data)
        cfg = json.loads(Path(config).read_text())
        pair = HFPair(cfg, str(Path(cfg["output"]) / "adapter"))
        val = stratified_take(
            read_jsonl(Path(data) / "validation.jsonl"), cfg.get("validation_limit"), cfg["seed"]
        )
        scores = score(predictions(HFView(pair, True), val))
        results.append(
            {
                "config": config,
                "adapter": str(Path(cfg["output"]) / "adapter"),
                "validation": scores,
                "duration_seconds": manifest["duration_seconds"],
            }
        )
        del pair
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    selected = max(results, key=lambda r: r["validation"]["macro_f1"])
    report = {
        "selection_metric": "validation_macro_f1",
        "test_used_for_selection": False,
        "runs": results,
        "selected": selected,
    }
    write_json(output, report)
    return report
