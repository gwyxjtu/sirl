#!/usr/bin/env python3
"""LP structure metrics for benchmark eval (shared by online eval + offline rescore).

Computes the same fields logged during SIRL training rollouts:
    format_score, lp_score, pred_n_vars/cons/bin/int, pred_obj_type,
    plus ans_ok/code_ok derived from the ObjVal status protocol when provided.

lp_score modes:
    - "gt": compared against lp_ref / gt_stats
    - "heuristic": no GT available; uses lp_structure_reward fallback
    - "none": no LP file produced (no code / write failed)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REWARD_DIR = _SCRIPT_DIR.parent / "reward_func"
if str(_REWARD_DIR) not in sys.path:
    sys.path.insert(0, str(_REWARD_DIR))

from content_utils import (  # noqa: E402
    insert_lp_generation,
    lp_structure_reward,
    parse_lp_structure,
)

# Scalar fields persisted on each eval record / summary aggregates.
STRUCTURE_SCALAR_KEYS = (
    "ans_ok",
    "code_ok",
    "format_score",
    "lp_score",
    "pred_n_vars",
    "pred_n_cons",
    "pred_n_bin",
    "pred_n_int",
)


def format_reward(processed_str: str) -> float:
    """Match training `batch_score_gurobi.format_reward` (<python> tags)."""
    has_open = "<python>" in processed_str
    has_close = "</python>" in processed_str
    if has_open and has_close:
        return 1.0
    if has_open or has_close:
        return 0.5
    return 0.0


def empty_structure_metrics() -> dict[str, Any]:
    return {
        "ans_ok": 0.0,
        "code_ok": 0.0,
        "format_score": 0.0,
        "lp_score": 0.0,
        "lp_score_mode": "none",
        "has_gt_lp_ref": False,
        "pred_n_vars": 0,
        "pred_n_cons": 0,
        "pred_n_bin": 0,
        "pred_n_int": 0,
        "pred_obj_type": "",
        "lp_write_ok": False,
    }


def status_to_ans_code(status: int | None) -> tuple[float, float]:
    """Map eval status (1 PASS / 0 FAIL / 2 NO_CODE / 3 EXEC) → ans_ok, code_ok."""
    if status == 1:
        return 1.0, 1.0
    if status == 0:
        return 0.0, 1.0
    return 0.0, 0.0


def extract_gt_stats(item: dict | None) -> dict | None:
    """Pull lp_ref / flat structure stats from a dataset item."""
    if not item or not isinstance(item, dict):
        return None
    if isinstance(item.get("lp_ref"), dict) and item["lp_ref"]:
        return dict(item["lp_ref"])
    ei = item.get("extra_info")
    if isinstance(ei, dict):
        if isinstance(ei.get("lp_ref"), dict) and ei["lp_ref"]:
            return dict(ei["lp_ref"])
        if "num_variables" in ei or "coeff_A" in ei or "objective_type" in ei:
            return dict(ei)
    # Flat top-level GT stats (some preprocessed dumps).
    if "num_variables" in item and "num_constraints" in item:
        keys = (
            "objective_type",
            "num_variables",
            "num_constraints",
            "num_binary",
            "num_integer",
            "has_quadratic",
            "coeff_A",
            "coeff_b",
        )
        return {k: item[k] for k in keys if k in item}
    return None


def load_lp_ref_index(path: str | Path | None) -> dict[str, dict]:
    """Load a sidecar lp_ref index.

    Supported line formats (jsonl):
        {"idx": 0, "lp_ref": {...}}
        {"id": "...", "lp_ref": {...}}
        {"idx": 0, "num_variables": ..., ...}   # flat stats as lp_ref

    Also accepts a .json object mapping str(idx)|id → stats or {"lp_ref": ...}.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    index: dict[str, dict] = {}

    def _store(key: Any, stats: dict):
        if key is None or not stats:
            return
        index[str(key)] = stats

    def _normalize_stats(obj: dict) -> dict | None:
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("lp_ref"), dict):
            return dict(obj["lp_ref"])
        if "num_variables" in obj or "objective_type" in obj or "coeff_A" in obj:
            # Drop non-stat bookkeeping keys if present.
            skip = {"idx", "id", "dataset", "question", "en_question", "en_answer"}
            return {k: v for k, v in obj.items() if k not in skip}
        return None

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                stats = _normalize_stats(obj)
                if stats is None:
                    continue
                _store(obj.get("idx"), stats)
                _store(obj.get("id"), stats)
        return index

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                stats = _normalize_stats(v) or (
                    dict(v["lp_ref"]) if isinstance(v.get("lp_ref"), dict) else None
                )
                _store(k, stats or {})
    elif isinstance(data, list):
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            stats = _normalize_stats(obj)
            if stats is None:
                continue
            _store(obj.get("idx", i), stats)
            _store(obj.get("id"), stats)
    return index


def resolve_lp_ref_path(
    lp_ref_dir: str | Path | None, dataset_name: str
) -> Path | None:
    """Find `{lp_ref_dir}/{dataset_name}.jsonl` or `.json`."""
    if not lp_ref_dir:
        return None
    root = Path(lp_ref_dir)
    for ext in (".jsonl", ".json"):
        cand = root / f"{dataset_name}{ext}"
        if cand.exists():
            return cand
    return None


def lookup_gt_stats(
    item: dict,
    idx: int,
    lp_ref_index: dict[str, dict] | None = None,
) -> dict | None:
    """Prefer in-item lp_ref; fall back to sidecar index by idx / id."""
    stats = extract_gt_stats(item)
    if stats:
        return stats
    if not lp_ref_index:
        return None
    for key in (str(idx), str(item.get("id", "")), str(item.get("idx", ""))):
        if key and key in lp_ref_index:
            return lp_ref_index[key]
    return None


def _extract_raw_code(output_text: str) -> str | None:
    """Extract Gurobi code without injecting print/LP helpers."""
    import re

    match = re.search(r"<python>(.*?)</python>", output_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "```" in code:
            inner = re.search(r"```python(.*?)```", code, re.DOTALL)
            if inner:
                code = inner.group(1).strip()
        return code
    match = re.search(r"```python(.*?)```", output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def write_lp_and_parse(
    output_text: str,
    timeout: int = 60,
    lp_dir: str | None = None,
) -> tuple[dict, bool, str | None]:
    """Write LP via skip_optimize injection; return (pred_stats, ok, lp_path)."""
    raw = _extract_raw_code(output_text)
    if not raw:
        return {}, False, None

    out_dir = lp_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    lp_path = os.path.join(out_dir, f"eval_lp_{uuid.uuid4().hex[:10]}.lp")
    code = insert_lp_generation(raw, lp_path, skip_optimize=True)
    if not code:
        return {}, False, None

    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return {}, False, lp_path
    except Exception:
        return {}, False, lp_path

    if not os.path.exists(lp_path):
        # Still try parse; usually missing. Keep stderr hint unused.
        _ = proc.returncode
        return {}, False, lp_path

    try:
        stats = parse_lp_structure(lp_path)
    except Exception:
        stats = {}
    ok = bool(
        stats.get("objective_type")
        or stats.get("num_variables")
        or stats.get("num_constraints")
    )
    return stats, ok, lp_path


def score_output_structure(
    output_text: str,
    gt_stats: dict | None = None,
    status: int | None = None,
    timeout: int = 60,
    lp_dir: str | None = None,
    cleanup_lp: bool = True,
) -> dict[str, Any]:
    """Score format + LP structure for one model output string."""
    metrics = empty_structure_metrics()
    metrics["format_score"] = format_reward(output_text or "")
    ans_ok, code_ok = status_to_ans_code(status)
    metrics["ans_ok"] = ans_ok
    metrics["code_ok"] = code_ok
    metrics["has_gt_lp_ref"] = bool(gt_stats)

    if not output_text:
        return metrics

    pred_stats, lp_ok, lp_path = write_lp_and_parse(
        output_text, timeout=timeout, lp_dir=lp_dir
    )
    metrics["lp_write_ok"] = lp_ok
    if pred_stats:
        metrics["pred_n_vars"] = int(pred_stats.get("num_variables", 0) or 0)
        metrics["pred_n_cons"] = int(pred_stats.get("num_constraints", 0) or 0)
        metrics["pred_n_bin"] = int(pred_stats.get("num_binary", 0) or 0)
        metrics["pred_n_int"] = int(pred_stats.get("num_integer", 0) or 0)
        metrics["pred_obj_type"] = pred_stats.get("objective_type", "") or ""

    if lp_path and os.path.exists(lp_path):
        try:
            if gt_stats:
                metrics["lp_score"] = float(lp_structure_reward(lp_path, gt_stats))
                metrics["lp_score_mode"] = "gt"
            else:
                metrics["lp_score"] = float(lp_structure_reward(lp_path, None))
                metrics["lp_score_mode"] = "heuristic"
        except Exception:
            metrics["lp_score"] = 0.0
            metrics["lp_score_mode"] = "none"
        if cleanup_lp:
            try:
                os.remove(lp_path)
            except OSError:
                pass
    else:
        metrics["lp_score"] = 0.0
        metrics["lp_score_mode"] = "none"

    return metrics


def aggregate_structure_metrics(records: list[dict]) -> dict[str, Any]:
    """Mean / rate aggregates for summary.json."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "ans_ok_rate": 0.0,
            "code_ok_rate": 0.0,
            "format_score_mean": 0.0,
            "lp_score_mean": 0.0,
            "lp_score_gt_mean": 0.0,
            "lp_score_gt_n": 0,
            "lp_write_ok_rate": 0.0,
            "has_gt_lp_ref_rate": 0.0,
        }

    def _mean(key: str) -> float:
        return sum(float(r.get(key, 0.0) or 0.0) for r in records) / n

    gt_recs = [r for r in records if r.get("lp_score_mode") == "gt"]
    gt_n = len(gt_recs)
    return {
        "n": n,
        "ans_ok_rate": _mean("ans_ok"),
        "code_ok_rate": _mean("code_ok"),
        "format_score_mean": _mean("format_score"),
        "lp_score_mean": _mean("lp_score"),
        "lp_score_gt_mean": (
            sum(float(r.get("lp_score", 0.0) or 0.0) for r in gt_recs) / gt_n
            if gt_n
            else 0.0
        ),
        "lp_score_gt_n": gt_n,
        "lp_write_ok_rate": sum(1 for r in records if r.get("lp_write_ok")) / n,
        "has_gt_lp_ref_rate": sum(1 for r in records if r.get("has_gt_lp_ref")) / n,
        "pred_n_vars_mean": _mean("pred_n_vars"),
        "pred_n_cons_mean": _mean("pred_n_cons"),
        "pred_n_bin_mean": _mean("pred_n_bin"),
        "pred_n_int_mean": _mean("pred_n_int"),
    }
