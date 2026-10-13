"""Invalid predictions count as errors. Paired bootstrap samples abstracts, not tokens."""

import random
from collections import Counter
from .common import LABEL_NAMES, INVALID


def parse_label(raw):
    candidate = raw.strip().lower().strip('"').strip("'")
    return candidate if candidate in LABEL_NAMES else INVALID


def score(rows):
    if not rows:
        raise ValueError("Evaluation requires nonempty predictions")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")
    if any(r["gold"] not in LABEL_NAMES for r in rows):
        raise ValueError("Unknown gold label")
    per_class, confusion = {}, {g: {p: 0 for p in LABEL_NAMES + [INVALID]} for g in LABEL_NAMES}
    for r in rows:
        pred = r["prediction"] if r["prediction"] in LABEL_NAMES else INVALID
        confusion[r["gold"]][pred] += 1
    for label in LABEL_NAMES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in LABEL_NAMES)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    n = len(rows)
    correct = sum(r["prediction"] == r["gold"] for r in rows)
    invalid = sum(r["prediction"] not in LABEL_NAMES for r in rows)
    return {
        "n": n,
        "accuracy": correct / n,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 5,
        "invalid_rate": invalid / n,
        "per_class": per_class,
        "confusion": confusion,
    }


def percentile(values, q):
    if not values:
        raise ValueError("Cannot compute percentile of empty data")
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def compare(base, adapted, seed=42, bootstrap=2000):
    # Order-independent pairing, strict same examples AND text identity.
    score(base)
    score(adapted)
    b, a = {r["id"]: r for r in base}, {r["id"]: r for r in adapted}
    if b.keys() != a.keys():
        raise ValueError("Comparisons require exactly the same IDs")
    if any(b[k]["gold"] != a[k]["gold"] or b[k].get("text_sha256") != a[k].get("text_sha256") for k in b):
        raise ValueError("Gold labels or input hashes differ")
    deltas, changes = [], []
    for k in sorted(b):
        before, after = b[k]["prediction"] == b[k]["gold"], a[k]["prediction"] == a[k]["gold"]
        deltas.append(int(after) - int(before))
        changes.append(
            {
                "id": k,
                "gold": b[k]["gold"],
                "base": b[k]["prediction"],
                "adapted": a[k]["prediction"],
                "change": "improved"
                if after and not before
                else "regressed"
                if before and not after
                else "unchanged",
            }
        )
    from collections import defaultdict

    groups = defaultdict(list)
    for key, delta in zip(sorted(b), deltas):
        if b[key].get("group_id", key) != a[key].get("group_id", key):
            raise ValueError("Document groups differ")
        groups[b[key].get("group_id", key)].append(delta)
    grouped = [(sum(v), len(v)) for v in groups.values()]
    rng, n = random.Random(seed), len(deltas)
    samples = []
    for _ in range(bootstrap):
        selected = rng.choices(grouped, k=len(grouped))
        samples.append(sum(x[0] for x in selected) / sum(x[1] for x in selected))
    counts = Counter(r["change"] for r in changes)
    return {
        "base": score(base),
        "adapted": score(adapted),
        "accuracy_delta": sum(deltas) / n,
        "paired_accuracy_delta_ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "bootstrap_unit": "abstract",
        "bootstrap_groups": len(grouped),
        "bootstrap_seed": seed,
        "bootstrap_resamples": bootstrap,
        "changes": dict(counts),
        "examples": changes,
    }


def release_gate(report, min_gain=0.01, max_class_drop=0.05, retention=None, max_retention_drop=0.05):
    reasons = []
    if report["accuracy_delta"] < min_gain:
        reasons.append("Accuracy gain is below the configured minimum")
    if report["paired_accuracy_delta_ci95"][0] <= 0:
        reasons.append("Paired accuracy interval does not exclude zero")
    if report["adapted"]["invalid_rate"] > report["base"]["invalid_rate"]:
        reasons.append("Invalid outputs increased")
    for label in LABEL_NAMES:
        drop = report["base"]["per_class"][label]["recall"] - report["adapted"]["per_class"][label]["recall"]
        if drop > max_class_drop:
            reasons.append(f"Recall regression exceeds threshold: {label}")
    if retention is None:
        reasons.append("General capability retention has not been measured")
    elif retention["adapted_accuracy"] < retention["base_accuracy"] - max_retention_drop:
        reasons.append("General capability retention dropped beyond threshold")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "policy": {
            "min_accuracy_gain": min_gain,
            "max_class_recall_drop": max_class_drop,
            "max_retention_drop": max_retention_drop,
        },
    }
