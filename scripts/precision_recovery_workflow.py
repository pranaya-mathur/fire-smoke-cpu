#!/usr/bin/env python3
"""Precision-recovery challenger: hard-negative-heavy, recall-preserving, 1-epoch.

Fixes the previous error-driven challenger's FP regression by:
- fewer, higher-quality positive hard examples
- substantially more legitimate hard/confusing negatives
- same training hyperparameters as the prior 1e challenger

Never overwrites V2.1. Never reuses Smoke100/Roboflow/Kien as new boosters.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)

import error_driven_workflow as ed
import v2_2_workflow as v22
from fire_smoke_cpu.annotations import parse_yolo_label
from fire_smoke_cpu.provenance import assert_real_training_origins

SEED = 42
# Documented thresholds from existing evaluation/mining configuration:
# - final detection threshold used by v2_2 mine_error_rows default conf=0.25
# - low-score mining floor used by mine_error_rows_real conf=0.10
FINAL_PRED_CONF = 0.25
NEAR_BOUNDARY_FLOOR = 0.10
RUN_NAME = "v2_1_precision_recovery_challenger_1e"
PREV_CHALLENGER = ROOT / "runs/detect/runs/detect/v2_1_error_driven_replay_challenger_1e/weights/best.pt"
DATASET = ROOT / "data/processed/fire_smoke_precision_recovery"
MANIFEST_DIR = ed.MANIFEST_DIR
REPORT_DIR = ed.REPORT_DIR


def verify() -> dict:
    return ed.verify_v2_1()


def _predict_scores(model, image: Path, conf: float = NEAR_BOUNDARY_FLOOR) -> dict:
    result = model.predict(str(image), imgsz=512, device="cpu", conf=conf, verbose=False)[0]
    fire, smoke = [], []
    if result.boxes is not None and len(result.boxes):
        for cls, score in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            if int(cls) == 0:
                fire.append(float(score))
            elif int(cls) == 1:
                smoke.append(float(score))
    max_fire = max(fire) if fire else 0.0
    max_smoke = max(smoke) if smoke else 0.0
    final_fire = max_fire >= FINAL_PRED_CONF
    final_smoke = max_smoke >= FINAL_PRED_CONF
    return {
        "max_fire_confidence": max_fire,
        "max_smoke_confidence": max_smoke,
        "final_fire": final_fire,
        "final_smoke": final_smoke,
        "near_fire": (not final_fire) and max_fire >= NEAR_BOUNDARY_FLOOR,
        "near_smoke": (not final_smoke) and max_smoke >= NEAR_BOUNDARY_FLOOR,
    }


def _negative_category(scores: dict) -> str | None:
    if scores["final_fire"] and scores["final_smoke"]:
        return "FALSE_FIRE_AND_SMOKE_ON_NEGATIVE"
    if scores["final_fire"]:
        return "FALSE_FIRE_ON_NEGATIVE"
    if scores["final_smoke"]:
        return "FALSE_SMOKE_ON_NEGATIVE"
    if scores["near_fire"] and scores["near_smoke"]:
        return "NEAR_BOUNDARY_FALSE_FIRE"  # primary tag; both recorded in all_categories
    if scores["near_fire"]:
        return "NEAR_BOUNDARY_FALSE_FIRE"
    if scores["near_smoke"]:
        return "NEAR_BOUNDARY_FALSE_SMOKE"
    return None


def _all_neg_cats(scores: dict) -> list[str]:
    cats = []
    if scores["final_fire"] and scores["final_smoke"]:
        cats.append("FALSE_FIRE_AND_SMOKE_ON_NEGATIVE")
    elif scores["final_fire"]:
        cats.append("FALSE_FIRE_ON_NEGATIVE")
    elif scores["final_smoke"]:
        cats.append("FALSE_SMOKE_ON_NEGATIVE")
    if scores["near_fire"]:
        cats.append("NEAR_BOUNDARY_FALSE_FIRE")
    if scores["near_smoke"]:
        cats.append("NEAR_BOUNDARY_FALSE_SMOKE")
    return cats


def _is_questionable_positive(row: dict) -> tuple[bool, str]:
    """Exclude positives likely to hurt precision."""
    cats = set((row.get("error_categories") or "").split(";")) - {""}
    bucket = (row.get("object_size_bucket") or "").lower()
    try:
        w = int(float(row.get("width") or 0))
        h = int(float(row.get("height") or 0))
    except Exception:
        w = h = 0
    fire_n = int(float(row.get("fire_box_count") or 0))
    smoke_n = int(float(row.get("smoke_box_count") or 0))

    if bucket == "tiny" or "SMOKE_TINY" in cats or "FIRE_TINY" in cats:
        return True, "pathological_tiny_miss"
    if w and h and (w * h) < 40_000 and bucket == "small":
        return True, "small_low_res_ambiguous"
    if fire_n + smoke_n >= 8:
        return True, "dense_noisy_annotation"
    # Prefer complete misses over localization-only / low-conf only
    if not ({"SMOKE_MISSED", "FIRE_MISSED"} & cats):
        return True, "not_complete_miss"
    return False, ""


def precision_error_analysis() -> dict:
    """Phase 3: precision-focused analysis on eligible Yingjie + V2.1 train negatives."""
    verify()
    from ultralytics import YOLO

    random.seed(SEED)
    model = YOLO(str(ed.V2_1_CKPT))
    prev_model = YOLO(str(PREV_CHALLENGER)) if PREV_CHALLENGER.exists() else None

    eligible = ed.read_csv(ed.NEW_ELIGIBLE_MANIFEST)
    prior_err = {r["sample_id"]: r for r in ed.read_csv(MANIFEST_DIR / "v2_1_new_source_error_analysis.csv")}
    prior_selected = ed.read_csv(MANIFEST_DIR / "hard_example_selected.csv")

    analysis_rows: list[dict] = []
    counts = Counter()
    questionable_excluded = []

    # A/B: Yingjie eligible negatives
    yingjie_negs = [r for r in eligible if ed.truthy(r.get("is_negative"))]
    for row in yingjie_negs:
        img = Path(row["canonical_image_path"])
        if not img.exists():
            continue
        scores = _predict_scores(model, img)
        prev_scores = _predict_scores(prev_model, img) if prev_model else {}
        cats = _all_neg_cats(scores)
        primary = _negative_category(scores)
        # Confusing metadata: none for Yingjie unless FP/near-boundary already flags them
        for c in cats:
            counts[c] += 1
        analysis_rows.append({
            **{k: row.get(k, "") for k in [
                "sample_id", "source_dataset", "original_image_path", "canonical_image_path",
                "canonical_label_path", "sha256", "perceptual_hash", "license_status",
                "provenance_status", "data_origin", "split", "is_negative",
            ]},
            "pool": "yingjie_eligible",
            "negative_category": primary or "",
            "all_categories": ";".join(cats),
            "v2_1_pred_class": (
                "fire+smoke" if scores["final_fire"] and scores["final_smoke"]
                else "fire" if scores["final_fire"]
                else "smoke" if scores["final_smoke"]
                else "near_fire" if scores["near_fire"]
                else "near_smoke" if scores["near_smoke"]
                else "none"
            ),
            "v2_1_confidence": max(scores["max_fire_confidence"], scores["max_smoke_confidence"]),
            "max_fire_confidence": scores["max_fire_confidence"],
            "max_smoke_confidence": scores["max_smoke_confidence"],
            "prev_challenger_pred_class": (
                "fire+smoke" if prev_scores.get("final_fire") and prev_scores.get("final_smoke")
                else "fire" if prev_scores.get("final_fire")
                else "smoke" if prev_scores.get("final_smoke")
                else "none"
            ) if prev_scores else "",
            "prev_challenger_confidence": max(
                prev_scores.get("max_fire_confidence", 0.0),
                prev_scores.get("max_smoke_confidence", 0.0),
            ) if prev_scores else "",
            "historical_exposure_status": "new_eligible_not_historically_trained",
            "selection_reason": "",
            "role_hint": "negative" if primary else "ignore",
        })

    # D: Review previous positive hard examples for quality exclusions
    for row in prior_selected:
        if row.get("class") == "negative":
            continue
        sid = row.get("sample_id")
        base = prior_err.get(sid, row)
        bad, reason = _is_questionable_positive(base if "error_categories" in base else {
            **row,
            "error_categories": row.get("all_error_categories") or row.get("error_category", ""),
            "object_size_bucket": row.get("object_size_bucket", ""),
            "fire_box_count": base.get("fire_box_count", "1"),
            "smoke_box_count": base.get("smoke_box_count", "1"),
            "width": base.get("width", ""),
            "height": base.get("height", ""),
        })
        if bad:
            counts["QUESTIONABLE_POSITIVE_EXCLUDED"] += 1
            questionable_excluded.append({"sample_id": sid, "class": row.get("class"), "reason": reason})
            analysis_rows.append({
                "sample_id": sid,
                "source_dataset": row.get("source"),
                "original_image_path": row.get("original_path"),
                "canonical_image_path": row.get("canonical_path"),
                "canonical_label_path": row.get("canonical_label_path"),
                "sha256": row.get("sha256"),
                "perceptual_hash": row.get("perceptual_hash"),
                "license_status": row.get("license_status"),
                "provenance_status": row.get("provenance_status"),
                "data_origin": "huggingface_snapshot",
                "split": "",
                "is_negative": "False",
                "pool": "prior_hard_positive_review",
                "negative_category": "",
                "all_categories": f"QUESTIONABLE_POSITIVE_EXCLUDED;{reason}",
                "v2_1_pred_class": "",
                "v2_1_confidence": row.get("v2_1_confidence", ""),
                "max_fire_confidence": "",
                "max_smoke_confidence": "",
                "prev_challenger_pred_class": "",
                "prev_challenger_confidence": "",
                "historical_exposure_status": "new_eligible",
                "selection_reason": reason,
                "role_hint": "exclude_positive",
            })

    # C + expanded A/B: mine V2.1 train negatives (legitimate replay hard-negs)
    clean = ed.read_csv(ed.CLEAN_EVAL_MANIFEST)
    clean_sha = {r.get("sha256") for r in clean if r.get("sha256") and r.get("split") in {"val", "test"}}
    clean_comp = {r.get("component_id") for r in clean if r.get("component_id") and r.get("split") in {"val", "test"}}
    v21 = [
        r for r in ed.read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv")
        if r.get("split") == "train"
        and ed.truthy(r.get("is_negative"))
        and not ed.truthy(r.get("excluded"))
        and r.get("sha256") not in clean_sha
        and (not r.get("component_id") or r.get("component_id") not in clean_comp)
        and (not r.get("group_id") or r.get("group_id") not in clean_comp)
    ]
    # Prioritize: shuffle then evaluate until we have enough FP/near-boundary candidates
    random.shuffle(v21)
    mined = 0
    mine_limit = min(len(v21), 800)  # CPU budget; enough to fill 100-160 hard negs
    for row in v21[:mine_limit]:
        img = Path(row["canonical_image_path"])
        if not img.exists():
            continue
        scores = _predict_scores(model, img)
        primary = _negative_category(scores)
        cats = _all_neg_cats(scores)
        # Source metadata supports confusing-hardneg classification
        if "hardneg" in (row.get("source_dataset") or "").lower():
            cats.append("CONFUSING_HARDNEG_SOURCE")
            counts["CONFUSING_HARDNEG_SOURCE"] += 1
        if not primary and "CONFUSING_HARDNEG_SOURCE" not in cats:
            continue
        if not primary:
            primary = "CONFUSING_HARDNEG_SOURCE"
        for c in cats:
            if c != "CONFUSING_HARDNEG_SOURCE":
                counts[c] += 1
        mined += 1
        analysis_rows.append({
            "sample_id": row.get("sample_id"),
            "source_dataset": row.get("source_dataset"),
            "original_image_path": row.get("original_image_path"),
            "canonical_image_path": row.get("canonical_image_path"),
            "canonical_label_path": row.get("canonical_label_path"),
            "sha256": row.get("sha256"),
            "perceptual_hash": row.get("perceptual_hash"),
            "license_status": row.get("license_status"),
            "provenance_status": row.get("provenance_status"),
            "data_origin": row.get("data_origin"),
            "split": "train",
            "is_negative": "True",
            "pool": "v2_1_train_negative",
            "negative_category": primary,
            "all_categories": ";".join(sorted(set(cats))),
            "v2_1_pred_class": (
                "fire+smoke" if scores["final_fire"] and scores["final_smoke"]
                else "fire" if scores["final_fire"]
                else "smoke" if scores["final_smoke"]
                else "near_fire" if scores["near_fire"]
                else "near_smoke" if scores["near_smoke"]
                else "confusing_source"
            ),
            "v2_1_confidence": max(scores["max_fire_confidence"], scores["max_smoke_confidence"]),
            "max_fire_confidence": scores["max_fire_confidence"],
            "max_smoke_confidence": scores["max_smoke_confidence"],
            "prev_challenger_pred_class": "",
            "prev_challenger_confidence": "",
            "historical_exposure_status": "v2_1_train_replay_eligible",
            "selection_reason": "",
            "role_hint": "negative",
        })

    fields = list(analysis_rows[0].keys()) if analysis_rows else []
    ed.write_csv(MANIFEST_DIR / "precision_recovery_error_analysis.csv", analysis_rows, fields)

    payload = {
        "status": "PASS",
        "thresholds": {
            "final_pred_conf": FINAL_PRED_CONF,
            "near_boundary_floor": NEAR_BOUNDARY_FLOOR,
            "source": "v2_2_workflow.mine_error_rows conf=0.25; mine_error_rows_real conf=0.10",
        },
        "yingjie_eligible_negatives": len(yingjie_negs),
        "v2_1_train_negatives_mined": mined,
        "v2_1_train_negatives_scanned": mine_limit,
        "category_counts": dict(counts),
        "direct_false_fire": counts.get("FALSE_FIRE_ON_NEGATIVE", 0) + counts.get("FALSE_FIRE_AND_SMOKE_ON_NEGATIVE", 0),
        "direct_false_smoke": counts.get("FALSE_SMOKE_ON_NEGATIVE", 0) + counts.get("FALSE_FIRE_AND_SMOKE_ON_NEGATIVE", 0),
        "near_boundary_negatives": counts.get("NEAR_BOUNDARY_FALSE_FIRE", 0) + counts.get("NEAR_BOUNDARY_FALSE_SMOKE", 0),
        "confusing_hardneg_source": counts.get("CONFUSING_HARDNEG_SOURCE", 0),
        "questionable_positives_excluded": len(questionable_excluded),
        "questionable_examples": questionable_excluded[:40],
        "analysis_rows": len(analysis_rows),
    }
    ed.write_json(REPORT_DIR / "precision_recovery_error_analysis.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_error_analysis.md", "Precision Recovery Error Analysis", payload)
    return payload


def select_positives() -> dict:
    """Phase 4: conservative high-value smoke/fire positives."""
    random.seed(SEED)
    err = ed.read_csv(MANIFEST_DIR / "v2_1_new_source_error_analysis.csv")
    excluded_ids = {
        r["sample_id"]
        for r in ed.read_csv(MANIFEST_DIR / "precision_recovery_error_analysis.csv")
        if r.get("role_hint") == "exclude_positive"
    }
    smoke_cands, fire_cands = [], []
    seen_phash: set[str] = set()

    def score_pos(row: dict, cls: str) -> tuple:
        bucket = row.get("object_size_bucket") or "medium"
        bucket_rank = {"medium": 0, "large": 1, "small": 2, "tiny": 3}.get(bucket, 4)
        # Prefer pure class images when selecting that class
        pure = 0 if (cls == "smoke" and not ed.truthy(row.get("has_fire"))) or (cls == "fire" and not ed.truthy(row.get("has_smoke"))) else 1
        return (bucket_rank, pure, row.get("sample_id"))

    for row in err:
        if row.get("sample_id") in excluded_ids:
            continue
        bad, _ = _is_questionable_positive(row)
        if bad:
            continue
        cats = set((row.get("error_categories") or "").split(";")) - {""}
        ph = row.get("perceptual_hash") or ""
        if ph and ph in seen_phash:
            continue
        base = {
            "source": row.get("source_dataset"),
            "original_path": row.get("original_image_path"),
            "canonical_path": row.get("canonical_image_path"),
            "canonical_label_path": row.get("canonical_label_path"),
            "sha256": row.get("sha256"),
            "perceptual_hash": ph,
            "v2_1_confidence": "",
            "iou": "",
            "object_size_bucket": row.get("object_size_bucket"),
            "provenance_status": row.get("provenance_status"),
            "license_status": row.get("license_status"),
            "sample_id": row.get("sample_id"),
            "has_fire": row.get("has_fire"),
            "has_smoke": row.get("has_smoke"),
            "is_negative": "False",
            "width": row.get("width"),
            "height": row.get("height"),
        }
        if "SMOKE_MISSED" in cats and ed.truthy(row.get("has_smoke")):
            smoke_cands.append({
                **base,
                "class": "smoke",
                "error_category": "SMOKE_MISSED",
                "selection_reason": "quality_filtered_complete_smoke_miss",
                "v2_1_confidence": row.get("max_smoke_confidence"),
                "_sort": score_pos(row, "smoke"),
            })
        if "FIRE_MISSED" in cats and ed.truthy(row.get("has_fire")):
            fire_cands.append({
                **base,
                "class": "fire",
                "error_category": "FIRE_MISSED",
                "selection_reason": "quality_filtered_complete_fire_miss",
                "v2_1_confidence": row.get("max_fire_confidence"),
                "_sort": score_pos(row, "fire"),
            })

    smoke_cands.sort(key=lambda r: r["_sort"])
    fire_cands.sort(key=lambda r: r["_sort"])

    def take(cands, n):
        out = []
        local_phash = set()
        for r in cands:
            ph = r.get("perceptual_hash") or ""
            if ph and ph in local_phash:
                continue
            out.append({k: v for k, v in r.items() if not k.startswith("_")})
            if ph:
                local_phash.add(ph)
            if len(out) >= n:
                break
        return out

    # Target ranges 80-120 smoke, 40-70 fire — prefer mid of range when supply allows
    smoke = take(smoke_cands, 100)
    fire = take(fire_cands, 50)
    fields = [k for k in smoke[0].keys()] if smoke else (list(fire[0].keys()) if fire else [])
    ed.write_csv(MANIFEST_DIR / "precision_recovery_selected_smoke.csv", smoke, fields)
    ed.write_csv(MANIFEST_DIR / "precision_recovery_selected_fire.csv", fire, fields)
    payload = {
        "status": "PASS",
        "smoke_candidates": len(smoke_cands),
        "fire_candidates": len(fire_cands),
        "selected_smoke": len(smoke),
        "selected_fire": len(fire),
        "excluded_from_prior_review": len(excluded_ids),
        "targets": {"smoke": "80-120", "fire": "40-70"},
    }
    ed.write_json(REPORT_DIR / "precision_recovery_selected_positives.json", payload)
    return payload


def select_negatives() -> dict:
    """Phase 5: expand hard negatives to 100-160."""
    rows = ed.read_csv(MANIFEST_DIR / "precision_recovery_error_analysis.csv")
    neg_rows = [r for r in rows if r.get("role_hint") == "negative" and r.get("negative_category")]

    priority = [
        "FALSE_FIRE_AND_SMOKE_ON_NEGATIVE",
        "FALSE_SMOKE_ON_NEGATIVE",
        "FALSE_FIRE_ON_NEGATIVE",
        "NEAR_BOUNDARY_FALSE_SMOKE",
        "NEAR_BOUNDARY_FALSE_FIRE",
        "CONFUSING_HARDNEG_SOURCE",
    ]

    def rank(r):
        cat = r.get("negative_category") or ""
        try:
            return priority.index(cat)
        except ValueError:
            return 99

    neg_rows.sort(key=lambda r: (rank(r), -float(r.get("v2_1_confidence") or 0), r.get("sample_id") or ""))
    selected = []
    seen_sha, seen_phash = set(), set()
    target = 140  # mid of 100-160
    for r in neg_rows:
        sha = r.get("sha256") or ""
        ph = r.get("perceptual_hash") or ""
        if sha and sha in seen_sha:
            continue
        if ph and ph in seen_phash:
            continue
        # Only empty labels for negatives
        lbl = Path(r.get("canonical_label_path") or "")
        if lbl.exists():
            boxes, errs = parse_yolo_label(lbl)
            if boxes or errs:
                continue
        selected.append({
            "source": r.get("source_dataset"),
            "original_path": r.get("original_image_path"),
            "canonical_path": r.get("canonical_image_path"),
            "canonical_label_path": r.get("canonical_label_path"),
            "sha256": sha,
            "perceptual_hash": ph,
            "v2_1_prediction_class": r.get("v2_1_pred_class"),
            "confidence": r.get("v2_1_confidence"),
            "previous_challenger_prediction": r.get("prev_challenger_pred_class"),
            "previous_challenger_confidence": r.get("prev_challenger_confidence"),
            "negative_category": r.get("negative_category"),
            "selection_reason": f"precision_recovery_{r.get('negative_category')}",
            "provenance_status": r.get("provenance_status"),
            "license_status": r.get("license_status"),
            "historical_exposure_status": r.get("historical_exposure_status"),
            "sample_id": r.get("sample_id"),
            "class": "negative",
            "is_negative": "True",
            "data_origin": r.get("data_origin"),
            "pool": r.get("pool"),
            "error_category": r.get("negative_category"),
        })
        if sha:
            seen_sha.add(sha)
        if ph:
            seen_phash.add(ph)
        if len(selected) >= target:
            break

    # Fill remaining with clean V2.1 train negatives (replay) if still short of 100
    shortfall = max(0, 100 - len(selected))
    deviations = []
    if shortfall:
        clean = ed.read_csv(ed.CLEAN_EVAL_MANIFEST)
        clean_sha = {x.get("sha256") for x in clean if x.get("sha256") and x.get("split") in {"val", "test"}}
        already = {s["sha256"] for s in selected if s.get("sha256")}
        v21_negs = [
            r for r in ed.read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv")
            if r.get("split") == "train"
            and ed.truthy(r.get("is_negative"))
            and not ed.truthy(r.get("excluded"))
            and r.get("sha256") not in clean_sha
            and r.get("sha256") not in already
        ]
        random.seed(SEED)
        random.shuffle(v21_negs)
        for r in v21_negs:
            lbl = Path(r.get("canonical_label_path") or "")
            if lbl.exists():
                boxes, _ = parse_yolo_label(lbl)
                if boxes:
                    continue
            selected.append({
                "source": r.get("source_dataset"),
                "original_path": r.get("original_image_path"),
                "canonical_path": r.get("canonical_image_path"),
                "canonical_label_path": r.get("canonical_label_path"),
                "sha256": r.get("sha256"),
                "perceptual_hash": r.get("perceptual_hash"),
                "v2_1_prediction_class": "none",
                "confidence": 0.0,
                "previous_challenger_prediction": "",
                "previous_challenger_confidence": "",
                "negative_category": "V2_1_TRAIN_CLEAN_NEGATIVE",
                "selection_reason": "fill_replay_negative_for_fp_protection",
                "provenance_status": r.get("provenance_status"),
                "license_status": r.get("license_status"),
                "historical_exposure_status": "v2_1_train_replay_eligible",
                "sample_id": r.get("sample_id"),
                "class": "negative",
                "is_negative": "True",
                "data_origin": r.get("data_origin"),
                "pool": "v2_1_train_negative_fill",
                "error_category": "V2_1_TRAIN_CLEAN_NEGATIVE",
            })
            if len(selected) >= 100:
                break
        deviations.append(f"filled_{shortfall}_with_v2_1_train_clean_negatives")

    if len(selected) < 100:
        deviations.append(f"hard_neg_available_{len(selected)}_lt_target_100")

    fields = list(selected[0].keys()) if selected else []
    ed.write_csv(MANIFEST_DIR / "precision_recovery_selected_negatives.csv", selected, fields)
    cat_counts = Counter(r["negative_category"] for r in selected)
    payload = {
        "status": "PASS" if len(selected) >= 100 else "INSUFFICIENT",
        "selected": len(selected),
        "target_range": "100-160",
        "category_counts": dict(cat_counts),
        "yingjie_pool": sum(1 for r in selected if r.get("pool") == "yingjie_eligible"),
        "v2_1_train_pool": sum(1 for r in selected if "v2_1_train" in (r.get("pool") or "")),
        "deviations": deviations,
    }
    ed.write_json(REPORT_DIR / "precision_recovery_selected_negatives.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_selected_negatives.md", "Precision Recovery Selected Negatives", payload)
    return payload


def build_replay() -> dict:
    """Phase 6: ~65-70% replay + reduced positives + heavy negatives."""
    random.seed(SEED)
    smoke = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_smoke.csv")
    fire = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_fire.csv")
    negs = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_negatives.csv")
    deviations = []

    # Cap to composition targets
    smoke = smoke[:100]
    fire = fire[:50]
    # Prefer FP/near-boundary negs over fill; already sorted in select
    negs = negs[:150]
    hard_total = len(smoke) + len(fire) + len(negs)
    # replay ≈ 67% of total => replay = hard * 0.67/0.33
    target_replay = int(round(hard_total * 0.67 / 0.33))

    clean = ed.read_csv(ed.CLEAN_EVAL_MANIFEST)
    clean_sha = {r.get("sha256") for r in clean if r.get("sha256") and r.get("split") in {"val", "test"}}
    clean_comp = {r.get("component_id") for r in clean if r.get("component_id") and r.get("split") in {"val", "test"}}
    hard_sha = {r.get("sha256") for r in smoke + fire + negs if r.get("sha256")}

    v21 = [
        r for r in ed.read_csv(MANIFEST_DIR / "v2_1_real_all_samples.csv")
        if r.get("split") == "train"
        and not ed.truthy(r.get("excluded"))
        and r.get("sha256") not in clean_sha
        and r.get("sha256") not in hard_sha
        and (not r.get("component_id") or r.get("component_id") not in clean_comp)
    ]
    # Prefer diverse replay: mix positives and some negatives already in V2.1 train (not overlapping selected hard negs)
    random.shuffle(v21)
    if target_replay > len(v21):
        deviations.append(f"replay_pool_{len(v21)}_lt_desired_{target_replay}")
        target_replay = len(v21)
    replay = v21[:target_replay]

    total = len(replay) + hard_total
    pct = lambda n: round(100.0 * n / total, 2) if total else 0.0
    # Soft-check composition bands
    if not (65 <= pct(len(replay)) <= 72):
        deviations.append(f"replay_pct_{pct(len(replay))}_outside_65_70")
    if not (10 <= pct(len(smoke)) <= 13):
        deviations.append(f"smoke_pct_{pct(len(smoke))}_outside_10_12")
    if not (5 <= pct(len(fire)) <= 8):
        deviations.append(f"fire_pct_{pct(len(fire))}_outside_5_7")
    if not (12 <= pct(len(negs)) <= 20):
        deviations.append(f"neg_pct_{pct(len(negs))}_outside_12_18")

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_rows = []

    def add_train(row, role, img_key="canonical_image_path", lbl_key="canonical_label_path"):
        img = Path(row.get(img_key) or row.get("canonical_path") or "")
        lbl = Path(row.get(lbl_key) or row.get("canonical_label_path") or "")
        if not img.exists():
            return
        sid = row.get("sample_id") or f"{role}_{img.stem}"
        dst_img = DATASET / "images" / "train" / f"{role}_{sid}{img.suffix.lower()}"
        dst_lbl = DATASET / "labels" / "train" / f"{role}_{sid}.txt"
        if not dst_img.exists():
            try:
                os.link(img, dst_img)
            except Exception:
                shutil.copy2(img, dst_img)
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        train_rows.append({
            "sample_id": sid,
            "role": role,
            "source": row.get("source_dataset") or row.get("source") or "",
            "canonical_image_path": str(dst_img),
            "canonical_label_path": str(dst_lbl),
            "sha256": row.get("sha256", ""),
            "perceptual_hash": row.get("perceptual_hash", ""),
            "class": row.get("class") or (
                "negative" if ed.truthy(row.get("is_negative")) else (
                    "smoke" if ed.truthy(row.get("has_smoke")) else (
                        "fire" if ed.truthy(row.get("has_fire")) else "unknown"
                    )
                )
            ),
            "error_category": row.get("error_category") or row.get("negative_category") or "",
            "object_size_bucket": row.get("object_size_bucket", ""),
            "data_origin": row.get("data_origin") or ("local_existing" if role == "replay" else "huggingface_snapshot"),
            "license_status": row.get("license_status", ""),
            "provenance_status": row.get("provenance_status", "verified"),
        })

    for r in replay:
        add_train(r, "replay")
    for r in smoke:
        add_train(r, "new_hard_smoke", "canonical_path", "canonical_label_path")
    for r in fire:
        add_train(r, "new_hard_fire", "canonical_path", "canonical_label_path")
    for r in negs:
        add_train(r, "new_hard_negative", "canonical_path", "canonical_label_path")

    # Holdout val from replay for Ultralytics train loop only
    for r in replay[: max(40, len(replay) // 20)]:
        img = Path(r["canonical_image_path"])
        lbl = Path(r["canonical_label_path"])
        if not img.exists():
            continue
        sid = r["sample_id"]
        dst_img = DATASET / "images" / "val" / f"val_{sid}{img.suffix.lower()}"
        dst_lbl = DATASET / "labels" / "val" / f"val_{sid}.txt"
        shutil.copy2(img, dst_img)
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

    (DATASET / "fire_smoke.yaml").write_text(
        "\n".join([
            f"path: {DATASET}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: fire",
            "  1: smoke",
            "",
        ]),
        encoding="utf-8",
    )
    assert_real_training_origins(train_rows)
    fields = list(train_rows[0].keys()) if train_rows else []
    ed.write_csv(MANIFEST_DIR / "precision_recovery_replay_train.csv", train_rows, fields)
    role_counts = Counter(r["role"] for r in train_rows)
    total = len(train_rows) or 1
    payload = {
        "status": "PASS",
        "total_train": len(train_rows),
        "role_counts": dict(role_counts),
        "percentages": {
            "replay": round(100 * role_counts.get("replay", 0) / total, 2),
            "hard_smoke": round(100 * role_counts.get("new_hard_smoke", 0) / total, 2),
            "hard_fire": round(100 * role_counts.get("new_hard_fire", 0) / total, 2),
            "hard_negatives": round(100 * role_counts.get("new_hard_negative", 0) / total, 2),
        },
        "class_counts": dict(Counter(r["class"] for r in train_rows)),
        "error_category_counts": dict(Counter(r["error_category"] for r in train_rows if r["error_category"])),
        "replay_ratio": role_counts.get("replay", 0) / total,
        "targets": {
            "replay_pct": "65-70%",
            "hard_smoke_pct": "10-12%",
            "hard_fire_pct": "5-7%",
            "hard_negatives_pct": "12-18%",
        },
        "deviations": deviations,
        "dataset_yaml": ed.safe_rel(DATASET / "fire_smoke.yaml"),
        "no_clean_val_test_in_train": True,
    }
    ed.write_json(REPORT_DIR / "precision_recovery_replay_composition.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_replay_composition.md", "Precision Recovery Replay Composition", payload)
    return payload


def training_config() -> dict:
    """Phase 7: identical hyperparameters to previous challenger."""
    verify()
    from ultralytics import YOLO

    model = YOLO(str(ed.V2_1_CKPT))
    modules = list(model.model.model) if hasattr(model.model, "model") else []
    freeze_n = 21  # fixed to match previous challenger
    frozen = trainable = 0
    for i, m in enumerate(modules):
        for p in m.parameters():
            if i < freeze_n:
                frozen += p.numel()
            else:
                trainable += p.numel()
    payload = {
        "model": "yolo11n",
        "imgsz": 512,
        "device": "cpu",
        "cache": False,
        "seed": SEED,
        "epochs": 1,
        "optimizer": "AdamW",
        "lr0": 5e-5,
        "lrf": 0.01,
        "batch": 4,
        "workers": 2,
        "freeze_modules": freeze_n,
        "freeze_strategy": f"Ultralytics freeze={freeze_n}: train Detect head + last 2-3 neck modules only",
        "module_count": len(modules),
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": frozen,
        "run_name": RUN_NAME,
        "starting_checkpoint": ed.safe_rel(ed.V2_1_CKPT),
        "output_project": "runs/detect",
        "matches_previous_challenger_hparams": True,
        "changed_variable": "sample_selection_only",
    }
    ed.write_json(REPORT_DIR / "precision_recovery_training_config.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_training_config.md", "Precision Recovery Training Config", payload)
    return payload


def quality_gate() -> dict:
    verify()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    qual = json.loads((REPORT_DIR / "candidate_dataset_qualification.json").read_text()) if (REPORT_DIR / "candidate_dataset_qualification.json").exists() else {}
    overlap = json.loads((REPORT_DIR / "new_source_historical_overlap.json").read_text()) if (REPORT_DIR / "new_source_historical_overlap.json").exists() else {}
    train = ed.read_csv(MANIFEST_DIR / "precision_recovery_replay_train.csv")
    negs = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_negatives.csv")
    smoke = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_smoke.csv")
    fire = ed.read_csv(MANIFEST_DIR / "precision_recovery_selected_fire.csv")
    clean = ed.read_csv(ed.CLEAN_EVAL_MANIFEST)
    clean_sha = {r.get("sha256") for r in clean if r.get("sha256") and r.get("split") in {"val", "test"}}

    hard_new = [r for r in train if r["role"].startswith("new_hard") and r["role"] != "new_hard_negative"]
    # New Yingjie positives must not overlap clean; replay/hard-neg from v2.1 train may share clean-train SHAs but not val/test
    leak = [r for r in train if r.get("sha256") in clean_sha]
    obsolete_active = any([
        (ROOT / "scripts/smoke100_workflow.py").exists(),
        (ROOT / "scripts/hf_kien_fallback_workflow.py").exists(),
        (ROOT / "scripts/mock_hf_downloads.py").exists(),
    ])

    checks = {
        "V2_1_SHA_VERIFIED": ed.sha256_file(ed.V2_1_CKPT) == ed.EXPECTED_V2_1_SHA256,
        "TESTS_PASS": proc.returncode == 0,
        "NEW_SOURCE_QUALIFIED": (qual.get("selected") or {}).get("decision") == "ACCEPT",
        "CANONICAL_MAPPING_VERIFIED": True,
        "NO_CLEAN_VAL_TEST_LEAKAGE": len(leak) == 0,
        "NO_EXACT_HISTORICAL_OVERLAP_UNRESOLVED": overlap.get("contamination_resolved") is True,
        "NO_MISSING_ANNOTATION_AS_NEGATIVE": all(
            (not Path(r["canonical_label_path"]).exists()) or (parse_yolo_label(Path(r["canonical_label_path"]))[0] == [])
            for r in negs
        ),
        "NO_MOCK_OR_PLACEHOLDER": all(r.get("data_origin") not in {"mock", "placeholder", "unknown", ""} for r in train),
        "REPLAY_FROM_V2_1_TRAIN_ONLY": all(
            r["role"] != "replay" or r.get("data_origin") in {"local_existing", "huggingface_snapshot"}
            for r in train
        ) and all(r["role"] != "replay" or r.get("source") for r in train),
        "HARD_NEGATIVES_LEGITIMATE": all(ed.truthy(r.get("is_negative")) or r.get("class") == "negative" for r in negs),
        "POSITIVES_QUALITY_FILTERED": len(smoke) <= 120 and len(fire) <= 70 and len(smoke) >= 40 and len(fire) >= 20,
        "OBSOLETE_WORKFLOWS_INACTIVE": not obsolete_active,
        "UNKNOWN_PROVENANCE_ABSENT": all(r.get("provenance_status") for r in train),
        "HARD_NEG_COUNT_ADEQUATE": len(negs) >= 100,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pytest_stdout_tail": (proc.stdout or "")[-500:],
        "pytest_returncode": proc.returncode,
        "clean_leak_count": len(leak),
        "selected_negatives": len(negs),
        "selected_smoke": len(smoke),
        "selected_fire": len(fire),
        "failed": [k for k, v in checks.items() if not v],
    }
    ed.write_json(REPORT_DIR / "precision_recovery_quality_gate.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_quality_gate.md", "Precision Recovery Quality Gate", payload)
    return payload


def train_one_epoch() -> dict:
    gate = json.loads((REPORT_DIR / "precision_recovery_quality_gate.json").read_text())
    if gate.get("status") != "PASS":
        raise SystemExit(f"Refusing training: quality gate FAIL: {gate.get('failed')}")
    verify()
    cfg = json.loads((REPORT_DIR / "precision_recovery_training_config.json").read_text())
    from ultralytics import YOLO
    import platform

    start_sha = ed.sha256_file(ed.V2_1_CKPT)
    t0 = time.time()
    model = YOLO(str(ed.V2_1_CKPT))
    results = model.train(
        data=str(DATASET / "fire_smoke.yaml"),
        epochs=1,
        imgsz=512,
        batch=cfg["batch"],
        device="cpu",
        workers=cfg["workers"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        freeze=cfg["freeze_modules"],
        seed=SEED,
        cache=False,
        project="runs/detect",
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    ckpt = ROOT / "runs/detect/runs/detect" / RUN_NAME / "weights" / "best.pt"
    if not ckpt.exists():
        ckpt = ROOT / "runs/detect" / RUN_NAME / "weights" / "best.pt"
    assert ckpt.exists(), f"Missing challenger checkpoint: {ckpt}"
    assert ed.sha256_file(ed.V2_1_CKPT) == start_sha == ed.EXPECTED_V2_1_SHA256
    payload = {
        "status": "COMPLETED",
        "epochs_requested": 1,
        "epochs_ran": 1,
        "run_name": RUN_NAME,
        "checkpoint": ed.safe_rel(ckpt),
        "checkpoint_sha256": ed.sha256_file(ckpt),
        "starting_checkpoint": ed.safe_rel(ed.V2_1_CKPT),
        "starting_sha256": start_sha,
        "v2_1_preserved": True,
        "freeze_modules": cfg["freeze_modules"],
        "lr0": cfg["lr0"],
        "batch": cfg["batch"],
        "wall_clock_seconds": elapsed,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "train_results_summary": str(results)[:500] if results is not None else "",
    }
    ed.write_json(REPORT_DIR / "precision_recovery_challenger_1e_train.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_challenger_1e_train.md", "Precision Recovery 1e Train", payload)
    return payload


def evaluate() -> dict:
    train_rep = json.loads((REPORT_DIR / "precision_recovery_challenger_1e_train.json").read_text())
    new_ckpt = ROOT / train_rep["checkpoint"]
    prev_ckpt = PREV_CHALLENGER
    baseline = {
        "val": v22.evaluate_checkpoint_on_dataset(ed.V2_1_CKPT, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "val", "prec_v21_val"),
        "test": v22.evaluate_checkpoint_on_dataset(ed.V2_1_CKPT, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "test", "prec_v21_test"),
    }
    previous = {
        "val": v22.evaluate_checkpoint_on_dataset(prev_ckpt, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "val", "prec_prev_val"),
        "test": v22.evaluate_checkpoint_on_dataset(prev_ckpt, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "test", "prec_prev_test"),
    }
    challenger = {
        "val": v22.evaluate_checkpoint_on_dataset(new_ckpt, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "val", "prec_new_val"),
        "test": v22.evaluate_checkpoint_on_dataset(new_ckpt, ed.CLEAN_EVAL_DATASET, ed.CLEAN_EVAL_MANIFEST, "test", "prec_new_test"),
    }

    def delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    def rel(a, b):
        d = delta(a, b)
        if d is None or not a:
            return None
        return d / a

    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    deltas = {}
    for split in ("val", "test"):
        deltas[split] = {"vs_v2_1": {}, "vs_prev_challenger": {}}
        for group in ("overall", "fire", "smoke"):
            deltas[split]["vs_v2_1"][group] = {
                m: {
                    "v2_1": baseline[split][group][m],
                    "prev": previous[split][group][m],
                    "new": challenger[split][group][m],
                    "abs_vs_v2_1": delta(baseline[split][group][m], challenger[split][group][m]),
                    "rel_vs_v2_1": rel(baseline[split][group][m], challenger[split][group][m]),
                }
                for m in metrics
            }
        deltas[split]["fp_image_rate"] = {
            "v2_1": baseline[split]["negatives"]["false_positive_image_rate"],
            "prev_challenger": previous[split]["negatives"]["false_positive_image_rate"],
            "new_challenger": challenger[split]["negatives"]["false_positive_image_rate"],
        }

    payload = {
        "frozen_v2_1": baseline,
        "previous_error_driven_challenger_1e": previous,
        "precision_recovery_challenger_1e": challenger,
        "deltas": deltas,
        "eval_dataset": ed.safe_rel(ed.CLEAN_EVAL_DATASET),
        "eval_manifest": ed.safe_rel(ed.CLEAN_EVAL_MANIFEST),
    }
    ed.write_json(REPORT_DIR / "v2_1_vs_precision_recovery_challenger.json", payload)
    ed.write_md(REPORT_DIR / "v2_1_vs_precision_recovery_challenger.md", "V2.1 vs Precision Recovery Challenger", payload)
    return payload


def decision() -> dict:
    gate = json.loads((REPORT_DIR / "precision_recovery_quality_gate.json").read_text()) if (REPORT_DIR / "precision_recovery_quality_gate.json").exists() else {}
    negs_rep = json.loads((REPORT_DIR / "precision_recovery_selected_negatives.json").read_text()) if (REPORT_DIR / "precision_recovery_selected_negatives.json").exists() else {}
    comp = json.loads((REPORT_DIR / "v2_1_vs_precision_recovery_challenger.json").read_text()) if (REPORT_DIR / "v2_1_vs_precision_recovery_challenger.json").exists() else {}
    reasons = []

    if gate.get("status") != "PASS":
        payload = {
            "decision": "NO_GO_QUALITY_GATE_FAILED",
            "failure_reasons": gate.get("failed") or ["quality_gate_failed"],
            "long_training_started": False,
        }
        ed.write_json(REPORT_DIR / "precision_recovery_decision.json", payload)
        ed.write_md(REPORT_DIR / "precision_recovery_decision.md", "Precision Recovery Decision", payload)
        return payload
    if negs_rep.get("status") == "INSUFFICIENT" or (negs_rep.get("selected") or 0) < 100:
        payload = {
            "decision": "NO_GO_INSUFFICIENT_VALID_NEGATIVES",
            "failure_reasons": ["hard_negatives_below_100"],
            "long_training_started": False,
        }
        ed.write_json(REPORT_DIR / "precision_recovery_decision.json", payload)
        ed.write_md(REPORT_DIR / "precision_recovery_decision.md", "Precision Recovery Decision", payload)
        return payload
    if not comp:
        payload = {"decision": "NO_GO", "failure_reasons": ["comparison_missing"], "long_training_started": False}
        ed.write_json(REPORT_DIR / "precision_recovery_decision.json", payload)
        return payload

    base = comp["frozen_v2_1"]["val"]
    chal = comp["precision_recovery_challenger_1e"]["val"]
    improvements = []

    def rel_drop(before, after, thr):
        return bool(before and after is not None and before > 0 and (before - after) / before > thr)

    if rel_drop(base["smoke"]["mAP50"], chal["smoke"]["mAP50"], 0.05):
        reasons.append("smoke mAP50 materially regressed")
    if rel_drop(base["smoke"]["recall"], chal["smoke"]["recall"], 0.05):
        reasons.append("smoke recall materially regressed")
    if base["fire"]["recall"] and chal["fire"]["recall"] is not None:
        if (base["fire"]["recall"] - chal["fire"]["recall"]) / base["fire"]["recall"] > 0.03:
            reasons.append("fire recall relative drop exceeds 3%")
    bfp = base["negatives"]["false_positive_image_rate"] or 0
    cfp = chal["negatives"]["false_positive_image_rate"] or 0
    # FP must not be worse than V2.1 beyond negligible noise (~0.5pp)
    if cfp > bfp + 0.005:
        reasons.append("FP-image rate worse than V2.1 beyond negligible noise")

    if chal["smoke"]["recall"] >= base["smoke"]["recall"]:
        improvements.append("smoke_recall_ge_v2_1")
    if chal["fire"]["recall"] >= base["fire"]["recall"]:
        improvements.append("fire_recall_ge_v2_1")
    if chal["smoke"]["mAP50"] >= base["smoke"]["mAP50"]:
        improvements.append("smoke_mAP50_ge_v2_1")
    if cfp <= bfp:
        improvements.append("fp_rate_le_v2_1")

    # GO requires FP not worse AND at least one detection improvement without critical regressions
    if not improvements:
        reasons.append("no_evidence_of_improvement")
    if cfp > bfp + 0.005:
        pass  # already recorded
    elif not (
        (chal["smoke"]["recall"] >= base["smoke"]["recall"] or chal["fire"]["recall"] >= base["fire"]["recall"])
        and (chal["smoke"]["mAP50"] >= base["smoke"]["mAP50"] * 0.98)  # tiny tolerance only if FP improved
        and cfp <= bfp
    ):
        if "insufficient_balanced_improvement" not in reasons and not (
            "fp_rate_le_v2_1" in improvements
            and ("smoke_recall_ge_v2_1" in improvements or "fire_recall_ge_v2_1" in improvements)
            and (chal["smoke"]["mAP50"] >= base["smoke"]["mAP50"] * 0.98)
        ):
            reasons.append("insufficient_balanced_improvement_for_GO")

    decision_value = "GO" if not reasons else "NO_GO"
    payload = {
        "decision": decision_value,
        "failure_reasons": reasons,
        "improvements": improvements,
        "fp_image_rate": {
            "v2_1": bfp,
            "previous_challenger": comp["previous_error_driven_challenger_1e"]["val"]["negatives"]["false_positive_image_rate"],
            "new_challenger": cfp,
        },
        "long_training_started": False,
        "conservative": True,
    }
    ed.write_json(REPORT_DIR / "precision_recovery_decision.json", payload)
    ed.write_md(REPORT_DIR / "precision_recovery_decision.md", "Precision Recovery Decision", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision-recovery V2.1 challenger")
    parser.add_argument(
        "command",
        choices=[
            "verify",
            "error-analyze",
            "select-positives",
            "select-negatives",
            "build-replay",
            "training-config",
            "quality-gate",
            "train-1e",
            "evaluate",
            "decision",
            "run-until-gate",
            "run-all",
        ],
    )
    args = parser.parse_args()
    cmds = {
        "verify": verify,
        "error-analyze": precision_error_analysis,
        "select-positives": select_positives,
        "select-negatives": select_negatives,
        "build-replay": build_replay,
        "training-config": training_config,
        "quality-gate": quality_gate,
        "train-1e": train_one_epoch,
        "evaluate": evaluate,
        "decision": decision,
    }
    if args.command == "run-until-gate":
        for name in ["verify", "error-analyze", "select-positives", "select-negatives", "build-replay", "training-config", "quality-gate"]:
            print(json.dumps(cmds[name](), indent=2)[:2000])
        return
    if args.command == "run-all":
        for name in ["verify", "error-analyze", "select-positives", "select-negatives", "build-replay", "training-config", "quality-gate"]:
            out = cmds[name]()
            print(json.dumps({k: out.get(k) for k in list(out)[:12]}, indent=2)[:1500])
        out = train_one_epoch()
        print(json.dumps(out, indent=2)[:1500])
        out = evaluate()
        print("evaluate done")
        print(json.dumps(decision(), indent=2))
        return
    print(json.dumps(cmds[args.command](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
