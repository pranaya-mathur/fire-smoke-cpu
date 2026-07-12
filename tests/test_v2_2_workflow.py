from pathlib import Path
import runpy


def load_workflow():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "scripts/v2_2_workflow.py"), run_name="not_main")


def test_dsu_collapses_transitive_near_duplicates():
    mod = load_workflow()
    dsu = mod["DSU"]()
    dsu.union("a", "b")
    dsu.union("b", "c")
    groups = [set(members) for members in dsu.groups().values()]
    assert {"a", "b", "c"} in groups


def test_hamming_hex_detects_near_pairs():
    mod = load_workflow()
    assert mod["hamming_hex"]("0000", "0001") == 1
    assert mod["hamming_hex"]("0000", "ffff") == 16


def test_exact_leakage_uses_sha_across_splits_not_label_conflict():
    mod = load_workflow()
    rows = [
        {"sample_id": "a", "sha256": "same", "component_id": "ca", "split": "train"},
        {"sample_id": "b", "sha256": "same", "component_id": "cb", "split": "val"},
    ]
    leaks = mod["leakage_counts"](rows)
    assert leaks["exact_cross_split_duplicates"] == 1


def test_near_leakage_uses_component_split_integrity():
    mod = load_workflow()
    rows = [
        {"sample_id": "a", "sha256": "sha_a", "component_id": "same_component", "split": "train"},
        {"sample_id": "b", "sha256": "sha_b", "component_id": "same_component", "split": "test"},
    ]
    leaks = mod["leakage_counts"](rows)
    assert leaks["near_duplicate_components_spanning_splits"] == 1


def test_area_bucket_boundaries():
    mod = load_workflow()
    assert mod["area_bucket"](0.009) == "tiny"
    assert mod["area_bucket"](0.01) == "small"
    assert mod["area_bucket"](0.05) == "medium"
    assert mod["area_bucket"](0.20) == "large"


def test_assign_splits_keeps_groups_together():
    mod = load_workflow()
    rows = [
        {"sample_id": "a", "group_id": "g1", "source_dataset": "medyoussef/fire-smoke-hardnegatives-int8"},
        {"sample_id": "b", "group_id": "g1", "source_dataset": "medyoussef/fire-smoke-hardnegatives-int8"},
        {"sample_id": "c", "group_id": "g2", "source_dataset": "LibreYOLO/smoke-uvylj"},
    ]
    assignment = mod["assign_splits"](rows)
    assert set(assignment) == {"g1", "g2"}
    assert assignment["g1"] in {"train", "val", "test"}


def test_train_v2_2_has_8_epoch_ceiling_and_quality_gate_guard():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    assert "epochs = 1 if smoke_test else 8" in text
    assert "Refusing to train: V2.2 quality gate" in text


def test_video_evaluator_has_safe_defaults_and_threshold_controls():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/evaluate_video.py").read_text(encoding="utf-8")
    assert "--inference-fps" in text
    assert "--fire-threshold" in text
    assert "--smoke-threshold" in text
    assert "--infer-ground-truth-from-filename" in text
    assert "READY_NO_LOCAL_VIDEOS" in text
