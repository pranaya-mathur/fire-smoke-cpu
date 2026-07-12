from pathlib import Path
import runpy


def load_smoke100():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "scripts/smoke100_workflow.py"), run_name="not_main")


def test_smoke100_download_reads_only_roboflow_env_and_redacts_key():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/smoke100_workflow.py").read_text(encoding="utf-8")
    start = text.index("def download_smoke100")
    end = text.index("def raw_split_for")
    body = text[start:end]
    assert 'os.environ.get("ROBOFLOW_API_KEY"' in body
    assert "api_key_printed" in body
    assert "<redacted>" in body
    assert "PEMutpJV3kSQZGdOtByj" not in text


def test_smoke100_canonicalizes_to_smoke_class_only():
    mod = load_smoke100()
    from fire_smoke_cpu.annotations import YoloBox

    boxes, errors = mod["canonicalize_boxes"]([YoloBox(0, 0.5, 0.5, 0.2, 0.2)], {0: "smoke"})
    assert not errors
    assert len(boxes) == 1
    assert boxes[0].class_id == 1


def test_smoke100_challenger_training_is_one_epoch_cpu_and_separate_run():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/smoke100_workflow.py").read_text(encoding="utf-8")
    start = text.index("def train_challenger")
    end = text.index("def compare_challenger")
    body = text[start:end]
    assert "epochs=1" in body
    assert 'device="cpu"' in body
    assert "cache=False" in body
    assert "smoke100_v2_1_challenger_1e" in body
    assert "Refusing Smoke100 challenger training: quality gate" in body


def test_smoke100_quality_gate_requires_training_only_and_tests():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/smoke100_workflow.py").read_text(encoding="utf-8")
    start = text.index("def quality_gate")
    end = text.index("def train_challenger")
    body = text[start:end]
    assert "SMOKE100_TRAINING_ONLY" in body
    assert "TESTS_PASS" in body
    assert "NO_CLEAN_EVAL_CONTAMINATION_BY_V2_1_TRAINING" in body


def test_smoke100_decision_never_starts_long_training():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/smoke100_workflow.py").read_text(encoding="utf-8")
    start = text.index("def decision")
    end = text.index("def main")
    body = text[start:end]
    assert '"long_training_started": False' in body
    assert '"main_training_ran": False' in body
    assert "fire recall relative drop exceeds 3%" in body
    assert "smoke mAP50 materially regressed" in body
