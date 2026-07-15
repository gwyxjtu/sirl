#!/usr/bin/env python3
"""Evaluate a HF model on test_data benchmarks (pass@1 ObjVal protocol).

Default prompt matches SIRL training `_shot` parquet: python-only + KEY GUROBI
RULES, no <think>/<model>. Optional --repair_rounds feeds execution errors back
to the model for one or more correction attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Clear proxies before Gurobi / network side effects
for k in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ[k] = ""
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
SIRL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIRL_DIR / "tools"))

from fix_prompts import DEFAULT_USER_TEMPLATE  # noqa: E402
from rule_prompt_utils import gurobi_prompt_temp  # noqa: E402
from utils import (  # noqa: E402
    change_variable_types,
    extract_code_block,
    extract_obj,
    load_jsonl,
)

# Exact system prompt from gurobi_examples_OR_*_shot.parquet (RULES 1–3 only).
SHOT_SYSTEM = (
    "You are a helpful Assistant with expertise in mathematical modeling and the Gurobi solver.\n"
    "Given an optimization problem, provide the Gurobi Python code to solve it.\n"
    "\n"
    "Your response MUST follow this structure exactly:\n"
    "\n"
    "<python>\n"
    "Provide the complete Gurobi Python code to implement the model.\n"
    "IMPORTANT: Always include `from gurobipy import *` at the top.\n"
    "Print the optimal objective value and optimal solution clearly.\n"
    "</python>\n"
    "\n"
    "--- KEY GUROBI RULES (must follow) ---\n"
    "1. Use model.ObjVal (capital O,V) — NOT objVal; use var.X (capital X) — NOT var.x.\n"
    "2. Binary: vtype=GRB.BINARY; Integer: vtype=GRB.INTEGER; Continuous: vtype=GRB.CONTINUOUS.\n"
    "3. Always check model.status == GRB.OPTIMAL before printing results.\n"
    "---\n"
    "\n"
    "Do NOT include any thinking steps or mathematical model sections.\n"
    "Only output the <python> code block."
)

DEFAULT_DATASETS = [
    "NL4OPT.jsonl",
    "MAMO_EasyLP_fixed.jsonl",
    "MAMO_ComplexLP_fixed.jsonl",
    "IndustryOR_fixedV2.json",
    "OptMATH_Bench_166.jsonl",
    "OptiBench.jsonl",
]

STATUS_MAP = {1: "PASS", 0: "FAIL", 2: "NO_CODE_FOUND", 3: "EXECUTION_ERROR"}


def load_dataset(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return load_jsonl(str(path))
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected JSON structure in {path}")
    except json.JSONDecodeError:
        return load_jsonl(str(path))


def _run_code(code_snippet: str, timeout: int = 200) -> subprocess.CompletedProcess | str:
    try:
        return subprocess.run(
            [sys.executable, "-c", code_snippet],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            env={
                **os.environ,
                "https_proxy": "",
                "HTTPS_PROXY": "",
                "http_proxy": "",
                "HTTP_PROXY": "",
            },
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def _traceback_line_nos(stderr: str) -> list[int]:
    """Line numbers from File \"<string>\", line N (python -c)."""
    return [int(n) for n in re.findall(r'File "<string>", line (\d+)', stderr)]


def _format_code_context(code_snippet: str, line_no: int, radius: int = 2) -> str:
    """Show numbered source lines around the failing line (1-indexed)."""
    lines = code_snippet.splitlines()
    if not lines or line_no < 1:
        return ""
    lo = max(1, line_no - radius)
    hi = min(len(lines), line_no + radius)
    out = []
    for i in range(lo, hi + 1):
        mark = ">>>" if i == line_no else "   "
        out.append(f"{mark} {i:4d} | {lines[i - 1]}")
    return "\n".join(out)


def _enrich_exec_feedback(code_snippet: str, stderr: str) -> str:
    """Build repair feedback with offending source line(s), not only raw traceback.

    Line numbers refer to the executed code block (after extract_code_block /
    insert_print), which is the same form shown back to the model on repair.
    """
    err = (stderr or "").strip()
    if len(err) > 1200:
        err = err[-1200:]

    parts = [
        "Your previous Gurobi code failed to execute.",
        "Traceback (tail):",
        "```",
        err,
        "```",
    ]

    line_nos = _traceback_line_nos(stderr or "")
    # Prefer the last frame in the user's code (often the real site).
    if line_nos and code_snippet:
        loc = line_nos[-1]
        ctx = _format_code_context(code_snippet, loc, radius=2)
        if ctx:
            parts.append(
                f"Failing location in your <python> code (around line {loc}):"
            )
            parts.append("```")
            parts.append(ctx)
            parts.append("```")
        # If traceback walks through genexpr / nested frames, also show earlier frames.
        uniq = []
        for n in line_nos:
            if n not in uniq:
                uniq.append(n)
        extras = [n for n in uniq[:-1] if abs(n - loc) > 2][-2:]
        for n in extras:
            ctx2 = _format_code_context(code_snippet, n, radius=1)
            if ctx2:
                parts.append(f"Also referenced (line {n}):")
                parts.append("```")
                parts.append(ctx2)
                parts.append("```")

    low = (stderr or "").lower()
    if "keyerror" in low:
        m = re.search(r"KeyError:\s*(.+)", stderr or "")
        missing = m.group(1).strip() if m else "?"
        parts.append(
            "KeyError diagnosis:\n"
            f"- Missing key: {missing}\n"
            "- Likely cause: loop / constraint indices do not match the index set "
            "used in addVars / dict keys (0-based vs 1-based, self-loops like "
            "(i,i), or city/name labels vs range indices).\n"
            "- Fix: define one explicit index set (e.g. cities = [...]), create "
            "variables only on that set (and edges only for i != j if needed), "
            "and iterate with the same set everywhere."
        )
    elif "nameerror" in low:
        m = re.search(r"NameError:\s*(.+)", stderr or "")
        parts.append(
            "NameError diagnosis:\n"
            f"- {m.group(1).strip() if m else 'undefined name'}\n"
            "- Import missing names (e.g. `from math import pi`) or define the "
            "variable before use."
        )
    elif "typeerror" in low and "abs" in low:
        parts.append(
            "TypeError diagnosis:\n"
            "- Do not call Python abs() on a Gurobi LinExpr. Use an auxiliary "
            "variable + model.addGenConstrAbs(aux, expr), or a linear reformulation."
        )
    elif "indexerror" in low:
        parts.append(
            "IndexError diagnosis:\n"
            "- A list/array index is out of range. Align range(...) and list "
            "lengths with the data tables in the problem statement."
        )

    parts.append(
        "Revise the modeling at the cited lines (and any mismatched index sets), "
        "then output a complete corrected <python> block."
    )
    fb = "\n".join(parts)
    if len(fb) > 3500:
        fb = fb[:3500] + "\n...(feedback truncated)..."
    return fb


def evaluate_output(result_str: str, item: dict, solver_name: str = "gurobi") -> dict:
    """Return status plus diagnostics for repair feedback.

    status: 1 PASS, 0 FAIL, 2 NO_CODE, 3 EXEC_ERROR
    """
    sub_answer = item.get("en_answer")
    sub_answer = (
        None
        if sub_answer == "No Best Solution" or "-9999" in str(sub_answer)
        else float(sub_answer)
    )

    code_snippet = extract_code_block(result_str, solver_name)
    if not code_snippet:
        return {
            "status": 2,
            "feedback": (
                "Your previous response did not contain a valid executable "
                "<python>...</python> Gurobi code block (or the block was incomplete)."
            ),
            "stderr": "",
            "stdout": "",
        }

    result = _run_code(code_snippet)
    if result == "TIMEOUT":
        # Match legacy check_result behavior on timeout.
        return {
            "status": 1 if sub_answer is None else 0,
            "feedback": "Code execution timed out.",
            "stderr": "TimeoutExpired",
            "stdout": "",
        }

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown execution error").strip()
        return {
            "status": 3,
            "feedback": _enrich_exec_feedback(code_snippet, err),
            "stderr": result.stderr or "",
            "stdout": result.stdout or "",
        }

    solver_result = extract_obj(result.stdout, solver_name)
    if (
        solver_result is not None
        and sub_answer is not None
        and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) <= 1e-6
    ):
        return {"status": 1, "feedback": "", "stderr": "", "stdout": result.stdout or ""}

    # Legacy: try flipping variable types on infeasible / wrong answer.
    if "nfeasible" in result.stdout or (
        solver_result is not None
        and sub_answer is not None
        and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) > 1e-6
    ):
        result_str2 = change_variable_types(result_str)
        if result_str2:
            code2 = extract_code_block(result_str2, solver_name)
            if code2:
                result2 = _run_code(code2)
                if result2 != "TIMEOUT" and result2.returncode == 0:
                    new_result = extract_obj(result2.stdout, solver_name)
                    if "nfeasible" not in result2.stdout:
                        if (
                            new_result is not None
                            and sub_answer is not None
                            and np.abs(new_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6
                        ):
                            return {
                                "status": 1,
                                "feedback": "",
                                "stderr": "",
                                "stdout": result2.stdout or "",
                            }
                        if new_result == sub_answer:
                            return {
                                "status": 1,
                                "feedback": "",
                                "stderr": "",
                                "stdout": result2.stdout or "",
                            }

    if "nfeasible" in result.stdout:
        status = 1 if sub_answer is None else 0
        fb = (
            "Your model was reported infeasible by Gurobi. "
            "Please fix constraints / variable bounds and output a corrected <python> block."
            if status == 0
            else ""
        )
        return {
            "status": status,
            "feedback": fb,
            "stderr": "",
            "stdout": result.stdout or "",
        }

    if solver_result is not None and sub_answer is not None:
        ok = np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6
        return {
            "status": 1 if ok else 0,
            "feedback": "",  # wrong ObjVal: do not reveal ground truth
            "stderr": "",
            "stdout": result.stdout or "",
        }
    return {
        "status": 1 if solver_result == sub_answer else 0,
        "feedback": "",
        "stderr": "",
        "stdout": result.stdout or "",
    }


def check_result(result_str: str, item: dict, solver_name: str = "gurobi") -> int:
    return evaluate_output(result_str, item, solver_name)["status"]


def _apply_chat(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_messages(item: dict, prompt_style: str = "shot") -> tuple[list[dict], bool]:
    question = item["en_question"]
    if prompt_style == "reproduce":
        system = gurobi_prompt_temp["system"].format(question=question).strip()
        user = gurobi_prompt_temp["user"].format(question=question).strip()
        enable_thinking = True
    else:
        system = SHOT_SYSTEM
        user = DEFAULT_USER_TEMPLATE.format(question=question)
        enable_thinking = False
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, enable_thinking


def build_prompt(tokenizer, item: dict, prompt_style: str = "shot") -> str:
    messages, enable_thinking = build_messages(item, prompt_style)
    return _apply_chat(tokenizer, messages, enable_thinking)


def build_repair_prompt(
    tokenizer,
    item: dict,
    prompt_style: str,
    prev_assistant: str,
    feedback: str,
) -> str:
    messages, enable_thinking = build_messages(item, prompt_style)
    # Prefer the code block only to keep chat within max_model_len.
    code = extract_code_block(prev_assistant, "gurobi")
    if code:
        prev_content = f"<python>\n{code}\n</python>"
    else:
        prev_content = prev_assistant[-3500:]
    messages.append({"role": "assistant", "content": prev_content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "Fix the issue at the cited failing lines (especially any KeyError "
                "index-set mismatch) and output ONLY a complete corrected "
                "<python>...</python> Gurobi code block. "
                "Do not include thinking or explanations."
            ),
        }
    )
    return _apply_chat(tokenizer, messages, enable_thinking)


def needs_repair(status: int, feedback: str, repair_on_infeasible: bool) -> bool:
    if status in (2, 3):  # no code / exec error
        return True
    if repair_on_infeasible and status == 0 and feedback:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "test_data"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--prompt",
        choices=["shot", "reproduce"],
        default="shot",
        help="shot = training _shot prompt (default); reproduce = old think/model",
    )
    parser.add_argument(
        "--repair_rounds",
        type=int,
        default=1,
        help="If >0, feed execution/no_code errors back to the model for N correction rounds",
    )
    parser.add_argument(
        "--repair_on_infeasible",
        action="store_true",
        help="Also repair FAIL cases when Gurobi reports infeasible (no GT leakage)",
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling; official SIRL README uses 0.95",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "execution.log"

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"model={args.model_path}")
    log(f"prompt={args.prompt}")
    log(f"repair_rounds={args.repair_rounds} repair_on_infeasible={args.repair_on_infeasible}")
    log(f"datasets={args.datasets}")
    log(f"Loading vLLM tp={args.tp} ...")
    t0 = time.time()
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</s>"],
        repetition_penalty=1.02,
    )
    log(f"Model ready in {time.time() - t0:.1f}s")

    summary = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "repair_rounds": args.repair_rounds,
        "repair_on_infeasible": args.repair_on_infeasible,
        "datasets": {},
    }

    for filename in args.datasets:
        path = Path(args.data_dir) / filename
        name = filename.replace(".jsonl", "").replace(".json", "")
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n=== {filename} ===")
        if not path.exists():
            log(f"MISSING {path}")
            continue
        data = load_dataset(path)
        log(f"loaded n={len(data)}")

        prompts = [build_prompt(tokenizer, item, args.prompt) for item in data]
        outputs = model.generate(prompts, sampling)
        texts = [o.outputs[0].text for o in outputs]

        evals = [evaluate_output(text, item, "gurobi") for text, item in zip(texts, data)]
        repair_used = [0] * len(data)

        for round_i in range(1, max(0, args.repair_rounds) + 1):
            need_idx = [
                i
                for i, ev in enumerate(evals)
                if needs_repair(ev["status"], ev.get("feedback", ""), args.repair_on_infeasible)
            ]
            if not need_idx:
                log(f"repair round {round_i}: nothing to repair")
                break
            log(f"repair round {round_i}: fixing {len(need_idx)} / {len(data)}")
            repair_prompts = [
                build_repair_prompt(
                    tokenizer,
                    data[i],
                    args.prompt,
                    texts[i],
                    evals[i]["feedback"]
                    or "Please fix your previous Gurobi code and output a corrected <python> block.",
                )
                for i in need_idx
            ]
            repair_outs = model.generate(repair_prompts, sampling)
            for j, i in enumerate(need_idx):
                new_text = repair_outs[j].outputs[0].text
                new_ev = evaluate_output(new_text, data[i], "gurobi")
                # Keep repair if better or equal priority: PASS > FAIL > EXEC > NO_CODE
                # Prefer any improvement in status rank; always keep if new is PASS.
                old_s, new_s = evals[i]["status"], new_ev["status"]
                rank = {1: 3, 0: 2, 3: 1, 2: 0}
                if rank.get(new_s, -1) >= rank.get(old_s, -1):
                    texts[i] = new_text
                    evals[i] = new_ev
                    repair_used[i] = round_i

        statuses = []
        records = []
        for idx, (text, item, ev) in enumerate(zip(texts, data, evals)):
            status = ev["status"]
            statuses.append(status)
            rec = {
                "idx": idx,
                "id": item.get("id", idx),
                "status": status,
                "status_name": STATUS_MAP.get(status, "UNKNOWN"),
                "en_answer": item.get("en_answer"),
                "repair_round": repair_used[idx],
            }
            records.append(rec)
            with (ds_dir / f"{idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **rec,
                        "question": item.get("en_question", "")[:2000],
                        "output": text,
                        "last_feedback": ev.get("feedback", "")[:2000],
                    },
                    f,
                    ensure_ascii=False,
                )

        counts = {k: int(statuses.count(k)) for k in (0, 1, 2, 3)}
        n = len(statuses)
        pass1 = counts[1] / n if n else 0.0
        exec_ok = (counts[0] + counts[1]) / n if n else 0.0
        n_repaired = sum(1 for r in repair_used if r > 0)
        n_repaired_pass = sum(
            1 for i, r in enumerate(repair_used) if r > 0 and statuses[i] == 1
        )
        ds_summary = {
            "n": n,
            "pass@1": pass1,
            "pass_count": counts[1],
            "fail_count": counts[0],
            "no_code": counts[2],
            "exec_error": counts[3],
            "exec_ok_rate": exec_ok,
            "counts": counts,
            "repair_attempted": n_repaired,
            "repair_became_pass": n_repaired_pass,
        }
        summary["datasets"][name] = ds_summary
        with (ds_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"records": records, **ds_summary}, f, indent=2, ensure_ascii=False)

        log(
            f"{name}: pass@1={pass1:.4f} ({counts[1]}/{n}) "
            f"fail={counts[0]} no_code={counts[2]} exec_err={counts[3]} "
            f"repair_tried={n_repaired} repair_pass={n_repaired_pass}"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
