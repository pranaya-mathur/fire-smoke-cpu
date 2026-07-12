# Smoke100 Preflight

Status: `PASS`

- Branch: `main`
- HEAD: `22d09848f381e469fadba752f2504fad1af14ad8`
- Frozen V2.1 checkpoint SHA matches expected: `True`
- Disk free: `120327639040` bytes
- Python: `3.12.13`
- Scripts inspected: `48`
- Tests inspected: `4`
- .gitignore protects env/raw/processed/runs/weights: `True`

## Notes

- API key source is `ROBOFLOW_API_KEY`; no key value was printed.
- Existing V2.2 workflow includes duplicate, near-duplicate component, historical V2.1 exposure, and clean eval quality gates.
- Abort is required if the checkpoint SHA does not match.
