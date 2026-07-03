#!/usr/bin/env python3
"""
离线预处理：为每条训练/验证样本生成 GT LP 结构统计，存入 extra_info.lp_ref。

用法：
    python tools/build_lp_ref_stats.py                            # train + test
    python tools/build_lp_ref_stats.py --only-train               # 只处理 train
    python tools/build_lp_ref_stats.py --only-test                # 只处理 test
    python tools/build_lp_ref_stats.py --dry-run                  # 预览，不写入
    python tools/build_lp_ref_stats.py --timeout 60               # 设置代码执行超时 (默认 60s)
    python tools/build_lp_ref_stats.py --no-overwrite              # 跳过已有 lp_ref 的样本
    python tools/build_lp_ref_stats.py --overwrite-src             # 直接覆盖原文件 (否则写 _lpref)

输出字段 (写入 extra_info.lp_ref):
    objective_type   — 'Minimize' / 'Maximize' / ''
    num_variables    — 变量数
    num_constraints  — 约束数
    num_binary       — 0-1 变量数
    num_integer      — 整数变量数 (不含 Binary)
    has_quadratic    — 是否有二次项
"""

import argparse
import os
import sys
import uuid
import tempfile
import subprocess
import traceback
from typing import Optional

import pandas as pd

# 确保能 import content_utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REWARD_DIR = os.path.join(_SCRIPT_DIR, "..", "reward_func")
sys.path.insert(0, _REWARD_DIR)

from content_utils import (
    extract_code_block,
    insert_lp_generation,
    parse_lp_structure,
)


def extract_ref_code(output_text: str) -> Optional[str]:
    """
    从 reference output 文本中提取 Gurobi 代码。

    支持格式：
        - <python>...</python>
        - ```python...```
        - 纯代码 (以 import/from 开头)
    """
    if output_text is None:
        return None

    # 尝试 <python> 标签
    import re
    pattern = r"<python>(.*?)</python>"
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # 处理 <python>```python...```</python> 嵌套 case
        if "```" in code:
            inner = re.search(r"```python(.*?)```", code, re.DOTALL)
            if inner:
                code = inner.group(1).strip()
        return code

    # 尝试 ```python``` 代码块
    pattern = r"```python(.*?)```"
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试纯代码 (从 import/from 开始)
    stripped = output_text.strip()
    if stripped.startswith("from ") or stripped.startswith("import "):
        return stripped

    return output_text.strip()


def extract_lp_stats(reference_code: str, timeout: int = 60) -> Optional[dict]:
    """
    执行 reference 代码 → 写出 LP → 解析统计。

    Returns:
        dict | None
    """
    # 先注入 LP 生成（跳过求解，只写 LP）
    lp_path = os.path.join(tempfile.gettempdir(), f"gt_lp_{uuid.uuid4().hex[:8]}.lp")
    code = insert_lp_generation(reference_code, lp_path, skip_optimize=True)
    if code is None:
        return None

    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                # 绕过代理，避免 Gurobi WLS TLS 错误
                "https_proxy": "",
                "HTTPS_PROXY": "",
                "http_proxy": "",
                "HTTP_PROXY": "",
                "ALL_PROXY": "",
                "all_proxy": "",
                "NO_PROXY": "*",
                "no_proxy": "*",
            },
        )

        if not os.path.exists(lp_path) or os.path.getsize(lp_path) == 0:
            return None

        stats = parse_lp_structure(lp_path)
        return {
            "objective_type": stats["objective_type"],
            "num_variables": stats["num_variables"],
            "num_constraints": stats["num_constraints"],
            "num_binary": stats["num_binary"],
            "num_integer": stats["num_integer"],
            "has_quadratic": stats["has_quadratic"],
        }
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        try:
            os.remove(lp_path)
        except OSError:
            pass


def _process_single(args_tuple):
    """Worker function: extract lp_ref for one row, using in-process Gurobi."""
    import importlib
    idx, output_text, _timeout = args_tuple
    ref_code = extract_ref_code(output_text)
    if ref_code is None:
        return idx, None

    # In-process execution: inject LP write, exec the code, parse LP
    lp_path = os.path.join(tempfile.gettempdir(), f"gt_lp_{idx}_{uuid.uuid4().hex[:6]}.lp")
    code = insert_lp_generation(ref_code, lp_path, skip_optimize=True)
    if code is None:
        return idx, None

    try:
        # Run the reference code (skip optimize — just build model and write LP)
        exec(code, {"__name__": "__main__"})
        if os.path.exists(lp_path) and os.path.getsize(lp_path) > 0:
            stats = parse_lp_structure(lp_path)
            return idx, {
                "objective_type": stats["objective_type"],
                "num_variables": stats["num_variables"],
                "num_constraints": stats["num_constraints"],
                "num_binary": stats["num_binary"],
                "num_integer": stats["num_integer"],
                "has_quadratic": stats["has_quadratic"],
            }
    except Exception:
        pass
    finally:
        try:
            os.remove(lp_path)
        except OSError:
            pass
    return idx, None


def process_one(label: str, path: str, args: argparse.Namespace):
    if not os.path.exists(path):
        print(f"[SKIP] {label}: {path} not found")
        return

    print(f"[READ] {label}: {path}")
    df = pd.read_parquet(path)
    n = len(df)

    # 确保 extra_info 列存在
    if "extra_info" not in df.columns:
        df["extra_info"] = [{} for _ in range(n)]

    # 准备任务列表
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = []
    skip_count = 0
    for i in range(n):
        extra = df.iloc[i]["extra_info"]
        if extra is None or (hasattr(extra, "item") and extra.item() is None):
            extra = {}
        if isinstance(extra, dict) and "lp_ref" in extra and args.no_overwrite:
            skip_count += 1
            continue
        output = df.iloc[i].get("output")
        tasks.append((i, output, args.timeout))

    if skip_count > 0:
        print(f"  Skipping {skip_count} rows with existing lp_ref (--no-overwrite)")

    if not tasks:
        print(f"  No new tasks — all rows already have lp_ref.\n")
        return

    print(f"  Processing {len(tasks)} rows with {args.workers} workers...")

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_single, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            idx, lp_ref = future.result()
            if lp_ref is not None:
                extra = df.at[idx, "extra_info"]
                if not isinstance(extra, dict):
                    extra = {}
                extra["lp_ref"] = lp_ref
                df.at[idx, "extra_info"] = extra
                success += 1
            else:
                failed += 1

            if success + failed <= 5 and lp_ref is not None:
                print(f"  [{idx}] OK: {lp_ref}")
            elif success + failed <= 5:
                print(f"  [{idx}] FAIL")

            if (success + failed) % 500 == 0:
                print(f"  ... {success + failed}/{len(tasks)} processed (ok={success}, fail={failed})")

    print(f"\n[DONE] {label}: {n} rows, ok={success}, skip={skip_count}, fail={failed}")

    if args.dry_run:
        print(f"  [DRY-RUN] Would write to {path}")
        return

    out_path = path if args.overwrite_src else path.replace(".parquet", "_lpref.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  [WRITE] -> {out_path}")
    if args.overwrite_src:
        print(f"  [WARN] Original file overwritten. Make sure you have a backup.")


def main():
    ap = argparse.ArgumentParser(description="Precompute GT LP structure stats for OR training/val data")
    ap.add_argument("--trainset", default="/root/llm/sirl/SIRLTrain/trainset/gurobi_examples_OR_train_fixed.parquet")
    ap.add_argument("--testset", default="/root/llm/sirl/SIRLTrain/trainset/gurobi_examples_OR_test_fixed.parquet")
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--only-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Preview without writing")
    ap.add_argument("--timeout", type=int, default=60, help="Code execution timeout in seconds")
    ap.add_argument("--no-overwrite", action="store_true", help="Skip samples that already have lp_ref")
    ap.add_argument("--overwrite-src", action="store_true", help="Overwrite source file (default: write _lpref.parquet)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    ap.add_argument("--trainset-path", default=None, help="Override trainset path")
    ap.add_argument("--testset-path", default=None, help="Override testset path")
    args = ap.parse_args()

    trainset = args.trainset_path or args.trainset
    testset = args.testset_path or args.testset

    targets = []
    if args.only_test:
        targets.append(("test", testset))
    elif args.only_train:
        targets.append(("train", trainset))
    else:
        targets = [("train", trainset), ("test", testset)]

    for label, path in targets:
        process_one(label, path, args)

    print("Done.")


if __name__ == "__main__":
    main()
