from __future__ import annotations

from dataclasses import dataclass


ALLOWED_DATA_ORIGINS = {
    "huggingface_snapshot",
    "local_existing",
    "roboflow_universe",
    "synthetic",
    "mock",
    "placeholder",
    "unknown",
}

REAL_TRAINING_ALLOWED_ORIGINS = {"huggingface_snapshot", "local_existing", "roboflow_universe"}
REAL_TRAINING_FORBIDDEN_ORIGINS = {"mock", "placeholder", "unknown"}


@dataclass(frozen=True)
class OriginDecision:
    ok: bool
    reason: str


def validate_data_origin(origin: str, *, real_training: bool = False) -> OriginDecision:
    normalized = (origin or "").strip()
    if normalized not in ALLOWED_DATA_ORIGINS:
        return OriginDecision(False, f"unrecognized_data_origin:{normalized or '<empty>'}")
    if real_training and normalized not in REAL_TRAINING_ALLOWED_ORIGINS:
        return OriginDecision(False, f"forbidden_real_training_origin:{normalized}")
    return OriginDecision(True, "ok")


def assert_real_training_origins(rows: list[dict]) -> None:
    failures = []
    for index, row in enumerate(rows, start=1):
        decision = validate_data_origin(str(row.get("data_origin", "")), real_training=True)
        if not decision.ok:
            failures.append(f"row_{index}:{row.get('sample_id', '<missing>')}:{decision.reason}")
    if failures:
        preview = "; ".join(failures[:10])
        extra = f" (+{len(failures) - 10} more)" if len(failures) > 10 else ""
        raise ValueError(f"real Dataset V2.1 contains forbidden sample origins: {preview}{extra}")
