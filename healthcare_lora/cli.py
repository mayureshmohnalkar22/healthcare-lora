import argparse
import json
from pathlib import Path
from .common import write_json


def main():
    p = argparse.ArgumentParser(description="Healthcare LoRA experiment pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("prepare")
    sub.add_parser("baseline")
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/local.json")
    sweep = sub.add_parser("sweep")
    sweep.add_argument(
        "--configs",
        nargs="+",
        default=["configs/rank-8.json", "configs/rank-16.json", "configs/rank-32.json"],
    )
    ev = sub.add_parser("evaluate")
    ev.add_argument("--backend", choices=["baseline", "hf"], default="baseline")
    ev.add_argument("--config", default="configs/local.json")
    ev.add_argument("--adapter")
    ev.add_argument("--dataset", default="data/processed/test.jsonl")
    ev.add_argument("--output", default="results/experiments/evaluation")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--retention", action="store_true")
    base_eval = sub.add_parser("base-evaluate")
    base_eval.add_argument("--config", default="configs/local.json")
    base_eval.add_argument("--dataset", default="data/processed/validation.jsonl")
    base_eval.add_argument("--output", default="results/experiments/base-baseline")
    base_eval.add_argument("--limit", type=int)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", default=8000, type=int)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--url", default="http://127.0.0.1:8000")
    bench.add_argument("--requests", default=20, type=int)
    bench.add_argument("--concurrency", default=1, type=int)
    j = sub.add_parser("judge")
    j.add_argument("--dataset", default="data/processed/test.jsonl")
    j.add_argument("--base", required=True)
    j.add_argument("--adapted", required=True)
    j.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    j.add_argument("--model", required=True)
    j.add_argument("--output", default="results/experiments/judge.json")
    args = p.parse_args()
    if args.command == "fetch":
        from .data import fetch

        result = fetch()
    elif args.command == "prepare":
        from .data import prepare

        result = prepare()
    elif args.command == "baseline":
        from .baseline import train_baseline

        result = train_baseline()
    elif args.command == "train":
        from .train import train

        result = train(args.config)
    elif args.command == "sweep":
        from .sweep import sweep

        result = sweep(args.configs)
    elif args.command == "evaluate":
        from .evaluate import evaluate, retention_check

        retention = None
        if args.backend == "hf":
            from .model import HFPair, HFView

            cfg = json.loads(Path(args.config).read_text())
            adapter = args.adapter or str(Path(cfg["output"]) / "adapter")
            pair = HFPair(cfg, adapter)
            first, second = HFView(pair), HFView(pair, True)
            if args.retention:
                retention = retention_check(pair)
            metadata = {"model": cfg["model"], "revision": cfg["revision"], "adapter": adapter}
        else:
            if args.retention:
                p.error("Retention is only available with --backend hf")
            from .baseline import KeywordBaseline, TfidfBaseline

            first, second, metadata = KeywordBaseline(), TfidfBaseline(), {}
        result = evaluate(
            first, second, args.dataset, args.output, args.limit, retention=retention, metadata=metadata
        )
    elif args.command == "base-evaluate":
        from .model import HFPair, HFView
        from .baseline import predictions
        from .common import read_jsonl, write_jsonl, file_hash
        from .data import stratified_take
        from .metrics import score

        cfg = json.loads(Path(args.config).read_text())
        pair = HFPair(cfg)
        rows = stratified_take(read_jsonl(args.dataset), args.limit, cfg["seed"])
        outputs = predictions(HFView(pair), rows)
        result = {
            "metrics": score(outputs),
            "evidence_kind": "real_model",
            "model": cfg["model"],
            "revision": cfg["revision"],
            "dataset_sha256": file_hash(args.dataset),
            "limit": args.limit,
        }
        write_jsonl(Path(args.output) / "predictions.jsonl", outputs)
        write_json(Path(args.output) / "summary.json", result)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("healthcare_lora.api:app", host="127.0.0.1", port=args.port)
        return
    elif args.command == "judge":
        from .judge import judge

        result = judge(args.dataset, args.base, args.adapted, args.endpoint, args.model, args.output)
    else:
        from .benchmark import benchmark

        result = benchmark(args.url, args.requests, args.concurrency)
        write_json("results/experiments/serving-benchmark.json", result)
    # Full results live in models/results; keep terminal summaries readable.
    if args.command in ["evaluate", "train", "baseline", "sweep"]:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ["examples", "source_hashes", "selected_example_ids", "config", "sweep"]
                },
                indent=2,
            )[:6000]
        )
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
