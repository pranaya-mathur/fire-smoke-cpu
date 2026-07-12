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


def test_repair_checkpoint_sha_constant_and_verifier_present():
    mod = load_workflow()
    assert mod["EXPECTED_V2_1_SHA256"] == "8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea"
    assert "verify_v2_1_checkpoint_sha" in mod


def test_real_mining_does_not_silently_fallback_to_model_none():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def mine_error_rows_real")
    end = text.index("def deduplicate_v2_2_repaired")
    body = text[start:end]
    assert "model = None" not in body
    assert "failed to load frozen V2.1 checkpoint" in body
    assert "REAL_IOU_VALUES_COMPUTED" in body


def test_repaired_split_requires_fire_smoke_validation_bucket():
    mod = load_workflow()
    rows = [
        {"split": "val", "has_fire": True, "has_smoke": True, "is_negative": False},
        {"split": "test", "has_fire": True, "has_smoke": True, "is_negative": False},
    ]
    gates = mod["split_balance_gates"](rows)
    assert not gates["VALIDATION_HAS_FIRE_SMOKE"]


def test_simple_ssim_identical_image(tmp_path: Path):
    mod = load_workflow()
    from PIL import Image

    image = tmp_path / "same.jpg"
    Image.new("RGB", (32, 32), "red").save(image)
    assert mod["simple_ssim"](image, image) > 0.99


def test_clean_eval_constants_and_commands_present():
    mod = load_workflow()
    assert mod["V2_2_CLEAN_DATASET"].name == "fire_smoke_v2_2_clean_eval"
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    for command in [
        "historical-preflight",
        "reconstruct-exposure",
        "propagate-exposure",
        "audit-current-contamination",
        "build-clean-eval",
        "quality-gate-clean",
        "smoke-train-clean",
        "decision-clean",
    ]:
        assert command in text


def test_historical_exposure_marks_only_train_split_seen():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def reconstruct_v2_1_historical_training_exposure")
    end = text.index("def historical_training_sets")
    body = text[start:end]
    assert '"seen_by_v2_1_training": split == "train"' in body
    assert "ABORT: ambiguous V2.1 split reconstruction" in body


def test_component_exposure_blocks_val_and_test():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def propagate_historical_exposure_components")
    end = text.index("def component_exposure_map")
    body = text[start:end]
    assert '"eligible_for_train": True' in body
    assert '"eligible_for_val": not exposed' in body
    assert '"eligible_for_test": not exposed' in body


def test_mined_fire_samples_are_forced_train_only():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def force_mined_hard_fire_train_only")
    end = text.index("def clean_eval_base_rows")
    body = text[start:end]
    assert "v2_1_error_mined_training_only" in body
    assert '"MINED_HARD_FIRE_SAMPLES_TRAIN_ONLY"' in body
    assert '"NO_MINED_FIRE_SAMPLE_IN_VAL"' in body
    assert '"NO_MINED_FIRE_SAMPLE_IN_TEST"' in body


def test_clean_eval_quality_gate_has_historical_contamination_gates():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def clean_eval_quality_gate")
    end = text.index("def frozen_v2_1_on_clean")
    body = text[start:end]
    for gate in [
        "NO_V2_1_TRAIN_SHA_IN_VAL",
        "NO_V2_1_TRAIN_SHA_IN_TEST",
        "NO_V2_1_TRAIN_COMPONENT_IN_VAL",
        "NO_V2_1_TRAIN_COMPONENT_IN_TEST",
        "NO_MINED_FIRE_SAMPLE_IN_VAL",
        "NO_MINED_FIRE_SAMPLE_IN_TEST",
    ]:
        assert gate in body


def test_clean_smoke_has_one_epoch_ceiling_and_lr():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def train_v2_2_clean_smoke")
    end = text.index("def compare_clean_one_epoch")
    body = text[start:end]
    assert "epochs=1" in body
    assert "lr0=0.0001" in body
    assert "smoke_test_v2_2_clean_eval_1e" in body


def test_clean_decision_blocks_main_on_no_go_and_checks_tests():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/v2_2_workflow.py").read_text(encoding="utf-8")
    start = text.index("def clean_main_training_decision")
    end = text.index("def frozen_baseline")
    body = text[start:end]
    assert '"decision": "GO" if not reasons else "NO_GO"' in body
    assert '"main_training_ran": False' in body
    assert "tests did not pass" in body
