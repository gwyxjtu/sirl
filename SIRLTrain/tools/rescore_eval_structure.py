#!/usr/bin/env python3
"""Offline: add lp / structure metrics to an existing eval_* directory.

Does NOT regenerate model outputs. Reads each `{dataset}/{idx:04d}.json`,
writes LP from the saved `output`, scores structure, and updates the record
(+ dataset summary.json + root summary.json).

Usage:
    python tools/rescore_eval_structure.py \
        --eval_dir /root/autodl-tmp/eval_step62 \
        --data_dir /root/llm/sirl/test_data \
        --lp_ref_dir /path/to/lp_ref_sidecars

    # dry-run one dataset
    python tools/rescore_eval_structure.py \
        --eval_dir /root/autodl-tmp/eval_step62 \
        --datasets MAMO_ComplexLP_fixed \
        --limit 5 --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))

from structure_eval_utils import (  # noqa: E402
    STRUCTURE_SCALAR_KEYS,
    aggregate_structure_metrics,
    load_lp_ref_index,
    lookup_gt_stats,
    resolve_lp_ref_path,
    score_output_structure,
)


def _load_dataset_items(data_dir: Path, dataset_name: str) -> list[dict]:
    """Load original benchmark items if present (for in-item lp_ref / id)."""
    if not data_dir:
        return []
    for ext in (".jsonl", ".json"):
        # Common naming: MAMO_ComplexLP_fixed.jsonl, IndustryOR_fixedV2.json
        cand = data_dir / f"{dataset_name}{ext}"
        if cand.exists():
            if ext == ".jsonl":
                items = []
                with cand.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
                return items
            with cand.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def _iter_record_files(ds_dir: Path) -> list[Path]:
    files = sorted(ds_dir.glob("*.json"))
    return [p for p in files if p.name != "summary.json" and p.stem.isdigit()]


def rescore_dataset(
    ds_dir: Path,
    data_dir: Path | None,
    lp_ref_dir: Path | None,
    timeout: int,
    limit: int | None,
    dry_run: bool,
    log,
) -> dict:
    name = ds_dir.name
    record_files = _iter_record_files(ds_dir)
    if limit is not None:
        record_files = record_files[:limit]
    log(f"=== {name}: {len(record_files)} records ===")

    items = _load_dataset_items(data_dir, name) if data_dir else []
    lp_ref_index: dict = {}
    if lp_ref_dir:
        ref_path = resolve_lp_ref_path(lp_ref_dir, name)
        if ref_path is not None:
            lp_ref_index = load_lp_ref_index(ref_path)
            log(f"  lp_ref: {ref_path} ({len(lp_ref_index)} keys)")
        else:
            log(f"  lp_ref: missing under {lp_ref_dir}")

    records = []
    for path in record_files:
        with path.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        idx = int(rec.get("idx", int(path.stem)))
        item = items[idx] if 0 <= idx < len(items) else {
            "id": rec.get("id", idx),
            "lp_ref": rec.get("lp_ref"),
            "extra_info": rec.get("extra_info"),
        }
        gt_stats = lookup_gt_stats(item, idx, lp_ref_index)
        struct = score_output_structure(
            rec.get("output") or "",
            gt_stats=gt_stats,
            status=rec.get("status"),
            timeout=timeout,
        )
        rec.update(struct)
        records.append(
            {
                "idx": idx,
                "id": rec.get("id", idx),
                "status": rec.get("status"),
                "status_name": rec.get("status_name"),
                "en_answer": rec.get("en_answer"),
                "repair_round": rec.get("repair_round", 0),
                **struct,
            }
        )
        if not dry_run:
            with path.open("w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)

    # Preserve existing pass@1 fields from prior summary when present.
    prev_summary = {}
    summary_path = ds_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            prev_summary = json.load(f)

    statuses = [int(r.get("status", -1)) for r in records]
    n = len(statuses)
    counts = {k: statuses.count(k) for k in (0, 1, 2, 3)}
    ds_summary = {
        "n": n,
        "pass@1": counts[1] / n if n else prev_summary.get("pass@1", 0.0),
        "pass_count": counts[1],
        "fail_count": counts[0],
        "no_code": counts[2],
        "exec_error": counts[3],
        "exec_ok_rate": (counts[0] + counts[1]) / n if n else 0.0,
        "counts": {str(k): v for k, v in counts.items()},
        "repair_attempted": prev_summary.get(
            "repair_attempted",
            sum(1 for r in records if int(r.get("repair_round") or 0) > 0),
        ),
        "repair_became_pass": prev_summary.get("repair_became_pass", 0),
    }
    ds_summary.update(aggregate_structure_metrics(records))
    ds_summary["structure_keys"] = list(STRUCTURE_SCALAR_KEYS)
    ds_summary["rescored_structure"] = True

    out_payload = {"records": records, **ds_summary}
    if not dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

    log(
        f"  pass@1={ds_summary['pass@1']:.4f} "
        f"ans_ok={ds_summary['ans_ok_rate']:.4f} "
        f"code_ok={ds_summary['code_ok_rate']:.4f} "
        f"lp_mean={ds_summary['lp_score_mean']:.4f} "
        f"lp_gt_n={ds_summary['lp_score_gt_n']} "
        f"lp_write_ok={ds_summary['lp_write_ok_rate']:.4f}"
    )
    return ds_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_dir",
        required=True,
        help="Existing eval output dir (e.g. /root/autodl-tmp/eval_step62)",
    )
    parser.add_argument(
        "--data_dir",
        default=str(_REPO_ROOT / "test_data"),
        help="Original benchmark dir (for ids / optional in-item lp_ref)",
    )
    parser.add_argument(
        "--lp_ref_dir",
        default="",
        help="Optional sidecars {dataset}.jsonl with lp_ref for gt-mode scoring",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset folder names; default = all subdirs with ####.json",
    )
    parser.add_argument("--structure_timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Max records per dataset")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Score and print aggregates but do not rewrite files",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"eval_dir not found: {eval_dir}")

    data_dir = Path(args.data_dir) if args.data_dir else None
    lp_ref_dir = Path(args.lp_ref_dir) if args.lp_ref_dir else None

    def log(msg: str):
        print(msg, flush=True)

    if args.datasets:
        ds_names = args.datasets
    else:
        ds_names = sorted(
            p.name
            for p in eval_dir.iterdir()
            if p.is_dir() and _iter_record_files(p)
        )

    log(f"eval_dir={eval_dir}")
    log(f"datasets={ds_names}")
    log(f"dry_run={args.dry_run} lp_ref_dir={lp_ref_dir or '(none)'}")

    root_summary = {}
    root_summary_path = eval_dir / "summary.json"
    if root_summary_path.exists():
        with root_summary_path.open("r", encoding="utf-8") as f:
            root_summary = json.load(f)
    root_summary.setdefault("datasets", {})
    root_summary["score_structure"] = True
    root_summary["lp_ref_dir"] = str(lp_ref_dir) if lp_ref_dir else None
    root_summary["rescored_structure"] = True

    for name in ds_names:
        ds_dir = eval_dir / name
        if not ds_dir.is_dir():
            log(f"SKIP missing {ds_dir}")
            continue
        ds_summary = rescore_dataset(
            ds_dir=ds_dir,
            data_dir=data_dir,
            lp_ref_dir=lp_ref_dir,
            timeout=args.structure_timeout,
            limit=args.limit,
            dry_run=args.dry_run,
            log=log,
        )
        # Store compact summary at root (without full records list).
        root_summary["datasets"][name] = {
            k: v for k, v in ds_summary.items() if k != "records"
        }

    if not args.dry_run:
        with root_summary_path.open("w", encoding="utf-8") as f:
            json.dump(root_summary, f, indent=2, ensure_ascii=False)
        log(f"Wrote {root_summary_path}")
    else:
        log("dry_run: root summary not written")


if __name__ == "__main__":
    main()
