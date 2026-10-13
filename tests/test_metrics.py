import pytest
from healthcare_lora.common import LABEL_NAMES
from healthcare_lora.metrics import parse_label, score, compare, release_gate


def rows(labels):
    return [
        {"id": str(i), "gold": g, "prediction": p, "text_sha256": str(i)} for i, (g, p) in enumerate(labels)
    ]


def test_invalid_output_is_not_silently_matched():
    assert parse_label("results because the abstract discusses cancer") == "invalid"
    assert parse_label("RESULTS") == "results"
    r = score(rows([(LABEL_NAMES[0], "invalid")]))
    assert r["accuracy"] == 0 and r["invalid_rate"] == 1
    assert r["per_class"][LABEL_NAMES[0]]["recall"] == 0


def test_macro_f1_penalizes_missing_classes():
    r = score(rows([(LABEL_NAMES[0], LABEL_NAMES[0])]))
    assert r["macro_f1"] == 0.2


def test_pairing_is_order_independent_and_counts_regressions():
    b = rows([(LABEL_NAMES[0], LABEL_NAMES[0]), (LABEL_NAMES[1], LABEL_NAMES[0])])
    a = rows([(LABEL_NAMES[0], LABEL_NAMES[1]), (LABEL_NAMES[1], LABEL_NAMES[1])])
    r = compare(b, list(reversed(a)), bootstrap=100)
    assert r["accuracy_delta"] == 0
    assert r["changes"]["improved"] == 1 and r["changes"]["regressed"] == 1


def test_mismatched_text_is_rejected():
    b = rows([(LABEL_NAMES[0], LABEL_NAMES[0])])
    a = [{**b[0], "text_sha256": "different"}]
    with pytest.raises(ValueError, match="hashes"):
        compare(b, a)


def test_duplicate_ids_rejected():
    b = rows([(LABEL_NAMES[0], LABEL_NAMES[0])])
    with pytest.raises(ValueError, match="Duplicate"):
        compare(b + b, b + b)


def test_missing_examples_rejected():
    b = rows([(LABEL_NAMES[0], LABEL_NAMES[0]), (LABEL_NAMES[1], LABEL_NAMES[1])])
    with pytest.raises(ValueError, match="same IDs"):
        compare(b, b[:1])


def test_empty_predictions_rejected():
    with pytest.raises(ValueError):
        score([])


def test_retention_required_for_release():
    b = rows([(label, LABEL_NAMES[0]) for label in LABEL_NAMES] * 10)
    a = [{**r, "prediction": r["gold"]} for r in b]
    r = compare(b, a, bootstrap=100)
    assert not release_gate(r)["pass"]
    assert release_gate(r, retention={"base_accuracy": 1, "adapted_accuracy": 1})["pass"]
    assert not release_gate(r, retention={"base_accuracy": 1, "adapted_accuracy": 0.5})["pass"]


def test_bootstrap_reproducible():
    b = rows([(label, LABEL_NAMES[0]) for label in LABEL_NAMES])
    a = [{**r, "prediction": r["gold"]} for r in b]
    assert compare(b, a, seed=7, bootstrap=30) == compare(b, a, seed=7, bootstrap=30)


def test_bootstrap_groups_correlated_sentences_by_abstract():
    b = rows([(label, LABEL_NAMES[0]) for label in LABEL_NAMES] * 2)
    for r in b:
        r["group_id"] = "shared-abstract"
    a = [{**r, "prediction": r["gold"]} for r in b]
    report = compare(b, a, bootstrap=20)
    assert report["bootstrap_groups"] == 1
    assert report["bootstrap_unit"] == "abstract"
    # A single abstract resamples as a unit, so all replicates are identical.
    assert report["paired_accuracy_delta_ci95"] == [0.8, 0.8]
