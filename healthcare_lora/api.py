import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .baseline import KeywordBaseline, TfidfBaseline
from .common import LABEL_NAMES


class CompareRequest(BaseModel):
    text: str = Field(min_length=10, max_length=20000)
    expected_label: str | None = None


def create_app(base=None, adapted=None, backend=None):
    lock = Lock()
    state = {"base": base, "adapted": adapted, "backend": backend}

    @asynccontextmanager
    async def lifespan(app):
        if state["base"] is None:
            mode = backend or os.getenv("HEALTHCARE_LORA_BACKEND", "baseline")
            if mode == "baseline":
                state.update(
                    base=KeywordBaseline(),
                    adapted=TfidfBaseline(os.getenv("HEALTHCARE_LORA_ARTIFACT", "models/tfidf.json")),
                    backend=mode,
                )
            elif mode == "hf":
                from .model import HFPair, HFView

                cfg = json.loads(Path(os.getenv("HEALTHCARE_LORA_CONFIG", "configs/local.json")).read_text())
                pair = HFPair(cfg, os.getenv("HEALTHCARE_LORA_ADAPTER", "models/local-run/adapter"))
                state.update(base=HFView(pair), adapted=HFView(pair, True), backend=mode)
            else:
                raise ValueError("Unknown backend; choose baseline or hf")
        yield

    app = FastAPI(title="Healthcare LoRA", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    def dashboard():
        return FileResponse(Path(__file__).parent / "web" / "index.html")

    @app.get("/bundled-report")
    def bundled_report():
        path = Path(__file__).parent / "web" / "baseline-report.json"
        if not path.exists():
            raise HTTPException(404, "No bundled benchmark")
        return FileResponse(path)

    @app.get("/health")
    def health():
        return {
            "ready": state["base"] is not None,
            "backend": state["backend"],
            "models": [getattr(state[k], "name", None) for k in ["base", "adapted"]],
            "evidence_kind": getattr(state["base"], "evidence_kind", None),
            "labels": LABEL_NAMES,
        }

    @app.post("/compare")
    def compare(request: CompareRequest):
        if not request.text.strip():
            raise HTTPException(422, "An abstract is required")
        if request.expected_label is not None and request.expected_label not in LABEL_NAMES:
            raise HTTPException(422, "Unknown expected label")
        # Reject excess concurrency rather than accumulating an unbounded model queue.
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Inference is busy; retry after the current request completes")
        try:
            results = {}
            for k in ["base", "adapted"]:
                model = state[k]
                if model is None:
                    raise HTTPException(503, "Model is not ready")
                result = model.predict(request.text)
                if request.expected_label is not None:
                    result["correct"] = result["prediction"] == request.expected_label
                results[k] = {"name": model.name, **result}
            return {
                "backend": state["backend"],
                "evidence_kind": state["base"].evidence_kind,
                "expected_label": request.expected_label,
                **results,
            }
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            lock.release()

    return app


app = create_app()
