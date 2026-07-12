from pathlib import Path
import runpy


def load_mod():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "scripts/precision_recovery_workflow.py"), run_name="not_main")


def test_precision_recovery_hparams_match_previous():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/precision_recovery_workflow.py").read_text(encoding="utf-8")
    assert "epochs=1" in text
    assert 'device="cpu"' in text
    assert "cache=False" in text
    assert "v2_1_precision_recovery_challenger_1e" in text
    assert "Refusing training: quality gate FAIL" in text
    assert "5e-5" in text or "0.00005" in text
    assert "freeze_n = 21" in text


def test_thresholds_documented_from_existing_eval():
    mod = load_mod()
    assert mod["FINAL_PRED_CONF"] == 0.25
    assert mod["NEAR_BOUNDARY_FLOOR"] == 0.10


def test_obsolete_workflows_still_archived():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "scripts/smoke100_workflow.py").exists()
    assert not (root / "scripts/hf_kien_fallback_workflow.py").exists()
    assert (root / "archive/obsolete_workflows/smoke100_workflow.py").exists()
