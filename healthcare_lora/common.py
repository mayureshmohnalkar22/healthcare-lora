import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

LABELS = {1: "background", 2: "objective", 3: "methods", 4: "results", 5: "conclusions"}
LABEL_NAMES = list(LABELS.values())
INVALID = "invalid"
PROMPT_VERSION = "pubmed-rct-sentence-role-v1"
SYSTEM = "You classify the rhetorical role of a sentence from a published medical research abstract. Return exactly one role identifier."


def prompt(text):
    return (
        "Classify this sentence from a medical research abstract into one role: "
        + ", ".join(LABEL_NAMES)
        + ".\nSentence:\n"
        + text
    )


def messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(text)}]


def normalize(text):
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def environment():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(), "git_commit": commit}
