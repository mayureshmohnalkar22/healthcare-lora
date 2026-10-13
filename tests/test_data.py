import pytest
from healthcare_lora.common import LABEL_NAMES
from healthcare_lora.data import assert_disjoint, deduplicate, stratified_take


def fixture_rows():
    return [{"id": str(i), "group_id": str(i), "label": LABEL_NAMES[i % 5]} for i in range(50)]


def test_content_leakage_detected():
    r = fixture_rows()
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": r[:10], "test": r[9:]})


def test_sampling_is_deterministic_and_covers_labels():
    r = fixture_rows()
    assert stratified_take(r, 10) == stratified_take(r, 10)
    assert set(x["label"] for x in stratified_take(r, 10)) == set(LABEL_NAMES)
    assert len(stratified_take(r, 10)) == 10


def test_conflicting_duplicate_rejected():
    r = fixture_rows()[0]
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate([r, {**r, "label": LABEL_NAMES[1]}])


def test_exact_duplicates_removed():
    r = fixture_rows()[0]
    cleaned, n = deduplicate([r, r])
    assert cleaned == [r] and n == 1


def test_small_sample_rejected():
    with pytest.raises(ValueError):
        stratified_take(fixture_rows(), 2)


def test_rct_parser_preserves_documents_and_positions(tmp_path):
    from healthcare_lora.data import load_rct

    path = tmp_path / "rct.txt"
    path.write_text(
        "###12345\nBACKGROUND\tPrevious research remains unclear.\nOBJECTIVE\tWe aimed to evaluate treatment.\n\n###67890\nRESULTS\tOutcomes improved.\n"
    )
    rows = load_rct(path, "train")
    assert [r["group_id"] for r in rows] == ["12345", "12345", "67890"]
    assert [r["sentence_index"] for r in rows] == [0, 1, 0]
    assert rows[2]["label"] == "results"


def test_rct_parser_rejects_unknown_roles(tmp_path):
    from healthcare_lora.data import load_rct

    path = tmp_path / "rct.txt"
    path.write_text("###12345\nUNKNOWN\tA sentence.\n")
    with pytest.raises(ValueError, match="Invalid role"):
        load_rct(path, "test")


def test_rct_parser_rejects_duplicate_document_ids(tmp_path):
    from healthcare_lora.data import load_rct

    path = tmp_path / "rct.txt"
    path.write_text("###12345\nRESULTS\tA sentence.\n###12345\nRESULTS\tAnother sentence.\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_rct(path, "test")
