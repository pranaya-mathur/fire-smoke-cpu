# fire-smoke-cpu

CPU-first SecureVU fire and smoke object-detection training repository.

## Frozen champion: V2.1

| Field | Value |
|---|---|
| Checkpoint | `runs/detect/runs/detect/yolo11n_v2_1_real_512_12e/weights/best.pt` |
| SHA256 | `8eda741d3741ee8b8094ee8244d1a276f0bf7ca41d5ee3afb73099095dab6aea` |
| Classes | `0 = fire`, `1 = smoke` (immutable) |
| Device | CPU-first |

Trusted baseline metrics: `reports/v2_1_frozen_baseline.json`.

### Previous challengers (do not revive blindly)

| Experiment | Decision |
|---|---|
| V2.2 (main / repaired / clean-eval) | **NO_GO** |
| Smoke100 / Roboflow | **NO_GO** (download blocked — do not retry) |
| Kien HF Indoor fallback | **NO_GO** (historical overlap + metric regressions) |
| Error-driven Yingjie hard-example 1e challenger | **NO_GO** (FP-image rate regressed; see `reports/error_driven_challenger_decision.json`) |
| Precision-recovery hard-negative-heavy 1e challenger | **NO_GO** (FP improved but smoke mAP50 regressed; see `reports/precision_recovery_decision.json`) |

## Active improvement workflow (error-driven)

Blind dataset merges are **prohibited**.

Improve V2.1 only through:

1. genuinely new external data (not Kien, Smoke100, Roboflow, or already-used sources);
2. source qualification + license/provenance gates;
3. historical-overlap blocking (SHA256 + perceptual near-duplicates);
4. running **frozen V2.1** on candidate data **before** any training;
5. selecting only hard examples where V2.1 fails or struggles;
6. replay-balanced fine-tuning (~65–70% V2.1 train replay);
7. conservative partial-freeze, **one-epoch** CPU challenger;
8. strict GO / NO_GO promotion on clean val/test.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Full error-driven pipeline (stops if quality gate fails)
python scripts/error_driven_workflow.py run-until-gate
# After PASS gate only:
python scripts/error_driven_workflow.py train-1e
python scripts/error_driven_workflow.py evaluate
python scripts/error_driven_workflow.py decision
```

Long training requires explicit future approval **after** a GO decision. Never overwrite the V2.1 checkpoint.

## Setup / environment

```bash
python scripts/detect_hardware.py
python scripts/check_environment.py
```

## Historical pipeline scripts (kept for provenance)

- `scripts/real_v2_1_workflow.py` — build of frozen real V2.1
- `scripts/v2_2_workflow.py` — historical V2.2 repair/clean-eval tooling (reuse for eval helpers)
- `archive/obsolete_workflows/` — Smoke100, Kien fallback, mock stubs (reference only; do not re-run)

## Tests

```bash
./.venv/bin/python -m pytest -q
```
