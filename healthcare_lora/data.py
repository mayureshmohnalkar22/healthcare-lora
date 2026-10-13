"""Parse upstream PubMed 20k RCT without redistributing its source abstracts."""

import itertools
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from .common import LABELS, digest, file_hash, normalize, write_json, write_jsonl

SOURCE_REPO = "https://github.com/Franck-Dernoncourt/pubmed-rct"
REVISION = "17ed2cb0590decfca0266add0c76f254f67232b4"
SUBSET = "PubMed_20k_RCT"
FILES = ["train.txt", "dev.txt", "test.txt"]
ROLE_MAP = {
    "BACKGROUND": "background",
    "OBJECTIVE": "objective",
    "METHODS": "methods",
    "RESULTS": "results",
    "CONCLUSIONS": "conclusions",
}


def fetch(root="data/raw"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/{REVISION}/{SUBSET}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "Healthcare-LoRA/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        (root / name).write_bytes(payload)
    manifest = {
        "source": SOURCE_REPO,
        "revision": REVISION,
        "subset": SUBSET,
        "license": "No blanket redistribution license; see upstream README and NLM copyright guidance",
        "sha256": {name: file_hash(root / name) for name in FILES},
    }
    write_json(root / "source_manifest.json", manifest)
    return manifest


def load_rct(path, split):
    rows, pmid, index = [], None, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("###"):
            pmid, index = line[3:].strip(), 0
            if not pmid.isdigit():
                raise ValueError("Invalid PMID")
            continue
        if pmid is None or "\t" not in line:
            raise ValueError("Sentence appeared without an abstract ID or tab separator")
        label, text = line.split("\t", 1)
        if label not in ROLE_MAP or not text.strip():
            raise ValueError("Invalid role or empty sentence")
        rows.append(
            {
                "id": f"{pmid}-{index:02d}",
                "group_id": pmid,
                "text": text.strip(),
                "label": ROLE_MAP[label],
                "origin_split": split,
                "sentence_index": index,
            }
        )
        index += 1
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate PMID/sentence ID")
    return rows


def deduplicate(rows):
    # Helper for imported single-label corpora; the RCT loader preserves complete documents.
    seen, clean, duplicates = {}, [], 0
    for r in rows:
        key = r["group_id"]
        if key in seen:
            if seen[key] != r["label"]:
                raise ValueError("Conflicting labels for identical normalized input")
            duplicates += 1
            continue
        seen[key] = r["label"]
        clean.append(r)
    return clean, duplicates


def assert_disjoint(splits):
    for (an, a), (bn, b) in itertools.combinations(splits.items(), 2):
        overlap = {r["group_id"] for r in a} & {r["group_id"] for r in b}
        if overlap:
            raise ValueError(f"Document leakage between {an} and {bn}: {len(overlap)} PMIDs")


def stratified_take(rows, n, seed=42):
    if n is None or n >= len(rows):
        return sorted(rows, key=lambda r: r["id"])
    if n < len(LABELS):
        raise ValueError("Sample size must cover all five labels")
    by_label = {label: [] for label in LABELS.values()}
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    picked = []
    while len(picked) < n:
        advanced = False
        for bucket in by_label.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop())
                advanced = True
        if not advanced:
            break
    return sorted(picked, key=lambda r: r["id"])


def prepare(raw="data/raw", output="data/processed", seed=42):
    raw, out = Path(raw), Path(output)
    source = json.loads((raw / "source_manifest.json").read_text())
    for name, expected in source["sha256"].items():
        if file_hash(raw / name) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    splits = {
        "train": load_rct(raw / "train.txt", "train"),
        "validation": load_rct(raw / "dev.txt", "validation"),
        "test": load_rct(raw / "test.txt", "test"),
    }
    assert_disjoint(splits)
    document_hashes, sentence_hashes = {}, {}
    for name, rows in splits.items():
        groups = defaultdict(list)
        for r in rows:
            groups[r["group_id"]].append(normalize(r["text"]))
        document_hashes[name] = {digest("\n".join(sentences)) for sentences in groups.values()}
        sentence_hashes[name] = {digest(normalize(r["text"])) for r in rows}
        write_jsonl(out / f"{name}.jsonl", rows)
    doc_overlap, sentence_overlap = {}, {}
    for a, b in itertools.combinations(splits, 2):
        doc_overlap[f"{a}:{b}"] = len(document_hashes[a] & document_hashes[b])
        sentence_overlap[f"{a}:{b}"] = len(sentence_hashes[a] & sentence_hashes[b])
    if any(doc_overlap.values()):
        raise ValueError("Exact full abstracts occur across splits; inspect upstream data before training")
    audit = {
        "source": source,
        "seed": seed,
        "split_policy": "official train/dev/test retained at abstract level",
        "pmid_overlap": 0,
        "full_abstract_content_overlap": doc_overlap,
        "shared_normalized_sentence_counts": sentence_overlap,
        "limitations": [
            "Shared boilerplate sentences are retained in the official benchmark",
            "Near-duplicate abstracts and base-model pretraining contamination are not ruled out",
            "This project uses independent sentence classification, not sequential context",
        ],
        "splits": {
            name: {
                "rows": len(rows),
                "abstracts": len({r["group_id"] for r in rows}),
                "classes": dict(Counter(r["label"] for r in rows)),
                "sha256": file_hash(out / f"{name}.jsonl"),
            }
            for name, rows in splits.items()
        },
    }
    write_json(out / "audit.json", audit)
    return audit
