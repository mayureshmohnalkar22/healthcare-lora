"""Lightweight, actual classical models. Neither baseline is a LoRA adapter."""

import json
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .common import digest, environment, file_hash, normalize, read_jsonl, write_json

KEYWORDS = {
    "background": ["known", "previous", "common", "remains", "unclear"],
    "objective": ["objective", "aim", "purpose", "investigate", "evaluate"],
    "methods": ["randomized", "assigned", "enrolled", "participants", "measured"],
    "results": ["significant", "increased", "decreased", "p =", "confidence interval"],
    "conclusions": ["conclude", "suggest", "should", "effective", "findings"],
}


class KeywordBaseline:
    name = "keyword-rules"
    evidence_kind = "classical_baseline"

    def predict(self, text):
        start = time.perf_counter()
        scores = {label: sum(text.lower().count(k) for k in keys) for label, keys in KEYWORDS.items()}
        label = max(scores, key=scores.get)
        return {
            "raw_output": label,
            "prediction": label,
            "confidence": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "output_tokens": None,
        }


class TfidfBaseline:
    name = "tfidf-logistic-regression"
    evidence_kind = "classical_baseline"

    def __init__(self, path="models/tfidf.json"):
        payload = json.loads(Path(path).read_text())
        payload["vectorizer_params"]["ngram_range"] = tuple(payload["vectorizer_params"]["ngram_range"])
        self.vectorizer = TfidfVectorizer(**payload["vectorizer_params"], vocabulary=payload["vocabulary"])
        # Establish fitted vocabulary and then replace IDF. No executable pickle is loaded.
        self.vectorizer.fit(["placeholder"])
        self.vectorizer.idf_ = np.asarray(payload["idf"])
        self.coef = np.asarray(payload["coef"])
        self.intercept = np.asarray(payload["intercept"])
        self.classes = payload["classes"]
        self.manifest = payload["manifest"]

    def predict(self, text):
        start = time.perf_counter()
        x = self.vectorizer.transform([text])
        logits = np.asarray(x @ self.coef.T).ravel() + self.intercept
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        i = int(probs.argmax())
        label = self.classes[i]
        return {
            "raw_output": label,
            "prediction": label,
            "confidence": float(probs[i]),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "output_tokens": None,
        }


def train_baseline(data="data/processed", output="models/tfidf.json", seed=42, train_limit=20000):
    data = Path(data)
    train, val = read_jsonl(data / "train.jsonl"), read_jsonl(data / "validation.jsonl")
    from .data import stratified_take

    original_train_count = len(train)
    train = stratified_take(train, train_limit, seed)
    params = {"ngram_range": (1, 2), "max_features": 30000, "sublinear_tf": True, "min_df": 2}
    vec = TfidfVectorizer(**params)
    start = time.perf_counter()
    x, xv = vec.fit_transform([r["text"] for r in train]), vec.transform([r["text"] for r in val])
    candidates = []
    for c in [0.5, 1.0, 2.0]:
        clf = LogisticRegression(C=c, max_iter=500, random_state=seed).fit(x, [r["label"] for r in train])
        from .metrics import score

        pred = clf.predict(xv)
        result = score([{"id": r["id"], "gold": r["label"], "prediction": p} for r, p in zip(val, pred)])
        candidates.append((result["macro_f1"], c, clf, result))
    best = max(candidates, key=lambda item: item[0])
    selected = best[2]
    manifest = {
        "evidence_kind": "classical_baseline",
        "environment": environment(),
        "seed": seed,
        "train_sha256": file_hash(data / "train.jsonl"),
        "validation_sha256": file_hash(data / "validation.jsonl"),
        "train_rows_used": len(train),
        "full_train_rows": original_train_count,
        "train_limit": train_limit,
        "selected_example_ids": [r["id"] for r in train],
        "selected_C": best[1],
        "selection_metric": "validation_macro_f1",
        "training_seconds": time.perf_counter() - start,
        "sweep": [{"C": c, "validation": result} for _, c, _, result in candidates],
    }
    payload = {
        "vectorizer_params": params,
        "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": selected.coef_.tolist(),
        "intercept": selected.intercept_.tolist(),
        "classes": selected.classes_.tolist(),
        "manifest": manifest,
    }
    write_json(output, payload)
    return manifest


def predictions(model, rows):
    result = []
    for r in rows:
        p = model.predict(r["text"])
        result.append(
            {
                "id": r["id"],
                "group_id": r["group_id"],
                "text_sha256": digest(normalize(r["text"])),
                "gold": r["label"],
                **p,
            }
        )
    return result
