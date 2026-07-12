from pathlib import Path
import runpy


def load_workflow():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "scripts/error_driven_workflow.py"), run_name="not_main")


def test_error_driven_never_overwrites_v2_1_and_is_one_epoch():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/error_driven_workflow.py").read_text(encoding="utf-8")
    assert "epochs=1" in text
    assert 'device="cpu"' in text
    assert "cache=False" in text
    assert "v2_1_error_driven_replay_challenger_1e" in text
    assert "EXPECTED_V2_1_SHA256" in text
    assert "Refusing training: quality gate FAIL" in text
    assert "long_training_started\": False" in text or '"long_training_started": False' in text


def test_blocked_sources_include_kien_and_smoke100():
    mod = load_workflow()
    blocked = {b.lower() for b in mod["BLOCKED_NEW_SOURCES"]}
    assert "kien" in blocked or "kienngyuen" in blocked
    assert "smoke100" in blocked
    assert "medyoussef" in blocked


def test_size_bucket_and_iou_helpers():
    mod = load_workflow()
    from fire_smoke_cpu.annotations import YoloBox

    tiny = YoloBox(1, 0.5, 0.5, 0.05, 0.05)
    assert mod["_size_bucket"](tiny) == "tiny"
    a = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
    b = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
    assert mod["_box_iou"](a, b) > 0.99


def test_obsolete_smoke100_not_in_scripts():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "scripts/smoke100_workflow.py").exists()
    assert (root / "archive/obsolete_workflows/smoke100_workflow.py").exists()
    assert not (root / "scripts/hf_kien_fallback_workflow.py").exists()
