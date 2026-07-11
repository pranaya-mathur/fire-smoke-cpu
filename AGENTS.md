# SecureVU Fire/Smoke Training Repository

This repository is CPU-first and reproducibility-first.

- Canonical classes are always `0 = fire`, `1 = smoke`.
- Raw downloaded datasets are immutable inputs and must not be deleted or rewritten.
- Negative images are preserved with empty YOLO label files.
- Full-image RAM caching is disabled for training.
- Commercial license status is tracked explicitly and remains review-required unless clear source evidence says otherwise.
- Training is gated on annotation validation, duplicate leakage checks, visual QC artifacts, and a one-epoch smoke test.
