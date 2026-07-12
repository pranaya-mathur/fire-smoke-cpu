# Obsolete / Abandoned Workflows

These scripts are retained for historical reference only.

They must **not** be re-run as part of the active SecureVU improvement pipeline.

| Script | Outcome | Why archived |
|---|---|---|
| `smoke100_workflow.py` + wrappers | `NO_GO` | Roboflow Smoke100 download blocked; do not retry |
| `hf_kien_fallback_workflow.py` | `NO_GO` | Reused historically exposed Kien data; metric regressions |
| `mock_hf_downloads.py` | superseded | Mock / sandbox path; forbidden for real training |
| `analyze_negative_candidates.py` | dead stub | Hardcoded mocked zeros |
| `visual_qc.py` | dead stub | Hardcoded mocked QC |
| `deduplicate_v2_candidates.py` | dead stub | Fake zero-duplicate report |
| `build_dataset_v2.py` | obsolete | Orchestrated mock dedupe stubs |
| `synthesize_v2_data.py` / `build_synthetic_v2.py` / `build_v2_1.py` | superseded | Synthetic V2/V2.1 replaced by real V2.1 |
| `evaluate_v0_vs_v2.py` | superseded | Narrower than v0/v2/v2.1 comparator |
| `parse_hf_metadata_from_md.py` | one-off | Discovery utility |

Active champion: **frozen V2.1**. Active improvement path: `scripts/error_driven_workflow.py`.
