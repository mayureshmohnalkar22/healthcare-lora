from pathlib import Path
from .baseline import predictions
from .common import PROMPT_VERSION, environment, file_hash, read_jsonl, write_json, write_jsonl
from .data import stratified_take
from .metrics import compare, percentile, release_gate


def evaluate(base, adapted, dataset, output, limit=None, seed=42, retention=None, metadata=None):
    rows = stratified_take(read_jsonl(dataset), limit, seed)
    before, after = predictions(base, rows), predictions(adapted, rows)
    report = compare(before, after, seed=seed)
    for name, pred in [("base", before), ("adapted", after)]:
        latencies = [r["latency_ms"] for r in pred]
        report[name]["latency_ms"] = {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)}
        report[name]["input_truncated_count"] = sum(r.get("input_truncated", False) for r in pred)
    report["provenance"] = {
        "evidence_kind": base.evidence_kind,
        "models": [base.name, adapted.name],
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": file_hash(dataset),
        "sample_limit": limit,
        "seed": seed,
        "environment": environment(),
        "metadata": metadata or {},
    }
    report["retention"] = retention
    report["release_gate"] = release_gate(report, retention=retention)
    if base.evidence_kind != "real_model":
        report["release_gate"] = {
            "pass": False,
            "reasons": ["Classical baselines cannot establish LoRA readiness"],
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "base.jsonl", before)
    write_jsonl(output / "adapted.jsonl", after)
    write_json(output / "comparison.json", report)
    render_report(report, output / "experiment.md")
    return report


def render_report(report, path):
    p = report["provenance"]
    lines = [
        "# Healthcare LoRA experiment report",
        "",
        f"Evidence: **{p['evidence_kind']}**",
        "",
        f"Models: {p['models'][0]} vs {p['models'][1]}",
        "",
        f"Examples: {report['base']['n']} · dataset SHA-256: `{p['dataset_sha256']}`",
        "",
        "| Metric | First model | Second model |",
        "|---|---:|---:|",
    ]
    for metric in ["accuracy", "macro_f1", "invalid_rate"]:
        lines.append(f"| {metric} | {report['base'][metric]:.4f} | {report['adapted'][metric]:.4f} |")
    lines += [
        "",
        f"Paired accuracy delta: {report['accuracy_delta']:.4f}",
        f"95% bootstrap interval: {report['paired_accuracy_delta_ci95']}",
        "",
        "## Per-class recall",
        "",
        "| Class | First | Second |",
        "|---|---:|---:|",
    ]
    for label in report["base"]["per_class"]:
        lines.append(
            f"| {label} | {report['base']['per_class'][label]['recall']:.4f} | {report['adapted']['per_class'][label]['recall']:.4f} |"
        )
    lines += ["", "## Gate decision", "", f"Pass: {report['release_gate']['pass']}"]
    lines += [f"- {reason}" for reason in report["release_gate"]["reasons"]]
    lines += [
        "",
        "## Interpretation",
        "",
        "When sample_limit is set, the evaluation is a subset of the upstream benchmark. "
        "Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. "
        "Clinical utility is not measured by this research classification task.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def retention_check(pair, path="data/evaluation/retention.jsonl"):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty retention suite")
    outputs = []
    for r in rows:
        item = {"id": r["id"], "expected": r["answer"]}
        for name, adapted in [("base", False), ("adapted", True)]:
            result = pair.generate([{"role": "user", "content": r["prompt"]}], adapted=adapted)
            raw = result["raw_output"].strip().casefold()
            item[name] = {
                "raw_output": result["raw_output"],
                "correct": raw in [a.casefold() for a in r["accepted"]],
            }
        outputs.append(item)
    return {
        "base_accuracy": sum(r["base"]["correct"] for r in outputs) / len(outputs),
        "adapted_accuracy": sum(r["adapted"]["correct"] for r in outputs) / len(outputs),
        "n": len(outputs),
        "suite_sha256": file_hash(path),
        "examples": outputs,
        "scope": "authored strict-answer/format smoke suite; not a standardized general-capability benchmark",
    }
