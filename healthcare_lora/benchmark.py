import time
from concurrent.futures import ThreadPoolExecutor
import httpx
from .metrics import percentile


def benchmark(url, requests=20, concurrency=1):
    if not 1 <= requests <= 1000 or not 1 <= concurrency <= 32:
        raise ValueError("requests must be 1-1000 and concurrency 1-32")
    text = "This randomized trial investigated coronary blood flow after myocardial infarction and compared cardiac outcomes between intervention groups."
    timings, errors = [], []

    def call(_):
        start = time.perf_counter()
        try:
            response = httpx.post(
                url.rstrip("/") + "/compare", json={"text": text}, timeout=120, trust_env=False
            )
            response.raise_for_status()
            return {"ms": (time.perf_counter() - start) * 1000, "body": response.json()}
        except httpx.HTTPError as e:
            return {"error": str(e)}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(call, range(requests)))
    duration = time.perf_counter() - start
    for r in results:
        if "error" in r:
            errors.append(r["error"])
        else:
            timings.append(r["ms"])
    total_tokens = sum(
        (r["body"][side].get("output_tokens") or 0)
        for r in results
        if "body" in r
        for side in ["base", "adapted"]
    )
    return {
        "requests": requests,
        "concurrency": concurrency,
        "successful": len(timings),
        "errors": errors,
        "end_to_end_ms": {"p50": percentile(timings, 0.5), "p95": percentile(timings, 0.95)}
        if timings
        else None,
        "successful_requests_per_second": len(timings) / duration,
        "output_tokens_per_second": total_tokens / duration,
        "measurement": "includes HTTP client setup; no warmup; both models run sequentially; saturation returns 429",
    }
