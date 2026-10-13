from fastapi.testclient import TestClient
from healthcare_lora.api import create_app
from healthcare_lora.baseline import KeywordBaseline


def client():
    return TestClient(create_app(KeywordBaseline(), KeywordBaseline(), "test"))


def test_compare_does_not_claim_accuracy_without_reference():
    with client() as c:
        r = c.post(
            "/compare", json={"text": "A study of myocardial infarction and cardiac outcomes in patients."}
        )
        assert r.status_code == 200
        assert r.json()["evidence_kind"] == "classical_baseline"
        assert "correct" not in r.json()["base"]


def test_invalid_reference_rejected():
    with client() as c:
        assert (
            c.post(
                "/compare",
                json={
                    "text": "A study of cardiac outcomes after treatment in patients.",
                    "expected_label": "unknown",
                },
            ).status_code
            == 422
        )


def test_short_and_oversized_inputs_rejected():
    with client() as c:
        assert c.post("/compare", json={"text": "hi"}).status_code == 422
        assert c.post("/compare", json={"text": "x" * 20001}).status_code == 422


def test_dashboard_and_health():
    with client() as c:
        assert c.get("/").status_code == 200
        assert c.get("/health").json()["ready"]


def test_reference_match():
    with client() as c:
        r = c.post(
            "/compare",
            json={
                "text": "Participants were randomized and assigned to two intervention groups.",
                "expected_label": "methods",
            },
        )
        assert r.json()["base"]["correct"]


def test_busy_inference_returns_429():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()

    class SlowModel(KeywordBaseline):
        def predict(self, text):
            entered.set()
            assert release.wait(5)
            return super().predict(text)

    with TestClient(create_app(SlowModel(), KeywordBaseline(), "test")) as c:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                c.post, "/compare", json={"text": "Participants were randomized to two groups."}
            )
            try:
                assert entered.wait(5)
                second = c.post("/compare", json={"text": "Participants were randomized to two groups."})
                assert second.status_code == 429
            finally:
                release.set()
            assert first.result().status_code == 200
