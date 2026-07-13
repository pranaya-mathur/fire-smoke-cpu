# Repository Cleanup Audit

- `archive/obsolete_workflows/hf_kien_fallback_workflow.py` — **ARCHIVE_REFERENCE_ONLY** — Kien reuse NO_GO — deps: no active imports — action: moved_to_archive
- `archive/obsolete_workflows/analyze_negative_candidates.py` — **REMOVE** — mocked stub — deps: unused — action: moved_to_archive
- `archive/obsolete_workflows/visual_qc.py` — **REMOVE** — mocked stub — deps: unused — action: moved_to_archive
- `archive/obsolete_workflows/deduplicate_v2_candidates.py` — **REMOVE** — fake zero-dup stub — deps: only build_dataset_v2 — action: moved_to_archive
- `archive/obsolete_workflows/mock_hf_downloads.py` — **ARCHIVE_REFERENCE_ONLY** — mock path forbidden for real training — deps: real_v2_1_workflow path updated — action: moved_to_archive
- `archive/obsolete_workflows/synthesize_v2_data.py` — **ARCHIVE_REFERENCE_ONLY** — synthetic superseded by real V2.1 — deps: unused — action: moved_to_archive
- `archive/obsolete_workflows/build_synthetic_v2.py` — **ARCHIVE_REFERENCE_ONLY** — synthetic superseded — deps: unused — action: moved_to_archive
- `archive/obsolete_workflows/build_v2_1.py` — **ARCHIVE_REFERENCE_ONLY** — synthetic V2.1 builder — deps: unused — action: moved_to_archive
- `archive/obsolete_workflows/build_dataset_v2.py` — **REMOVE** — orchestrated mock stubs — deps: unused — action: moved_to_archive
- `src/fire_smoke_cpu/hf_adapters/` — **REMOVE** — empty unimplemented scaffold — deps: no imports — action: deleted
- `scripts/error_driven_workflow.py` — **KEEP** — new error-driven pipeline — deps: n/a — action: created
- `scripts/v2_2_workflow.py` — **KEEP** — eval/dedupe utilities + historical V2.2 evidence — deps: tests + error_driven import — action: retained
- `scripts/real_v2_1_workflow.py` — **REFACTOR** — mock path reference updated — deps: ok — action: updated_mock_path
- `requirements.in` — **REFACTOR** — removed roboflow dependency — deps: ok — action: roboflow_removed
- `reports/*hf_kien*` — **KEEP** — historical NO_GO evidence — deps: n/a — action: retained
- `runs/.../yolo11n_v2_1_real_512_12e/weights/best.pt` — **KEEP** — frozen champion — deps: n/a — action: untouched
