# Repository Cleanup Final (Phase 16)

- pytest: **36 passed**
- decision: **NO_GO**

- `archive/obsolete_workflows/` — **ARCHIVE_REFERENCE_ONLY** — Smoke100/Kien/mock/synthetic one-offs; not active — retained_in_archive
- `scripts/error_driven_workflow.py` — **KEEP** — canonical error-driven pipeline — replay_filter_fixed_val_test_only
- `scripts/v2_2_workflow.py` — **KEEP** — eval/dedupe helpers + V2.2 provenance — retained
- `scripts/real_v2_1_workflow.py` — **KEEP** — V2.1 build provenance — retained
- `scripts/prepare_hf_indoor.py` — **KEEP** — historical Kien prep for registry only; not booster path — retained_for_provenance
- `requirements.in` — **REFACTOR** — roboflow removed earlier — unchanged_this_pass
- `README.md` — **REFACTOR** — documents V2.1 + error-driven rules — updated_with_challenger_nogo
- `reports/error_driven_* + v2_1_vs_*` — **KEEP** — challenger evidence — generated
- `runs/.../yolo11n_v2_1_real_512_12e/weights/best.pt` — **KEEP** — frozen champion SHA verified — untouched
- `runs/.../v2_1_error_driven_replay_challenger_1e/` — **KEEP** — NO_GO challenger evidence — retained_not_promoted

## Stale workflows eliminated
- Smoke100 download/retry
- Roboflow API downloads
- Kien fallback booster training
- blind full-dataset merges
- mock HF download path for real training
