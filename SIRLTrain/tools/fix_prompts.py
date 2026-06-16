#!/home/guo/anaconda3/bin/python
"""Patch system/user prompts in Gurobi OR parquet datasets.

Usage:
  python tools/fix_prompts.py                                    # default: fix train+test, output _fixed.parquet
  python tools/fix_prompts.py --dry-run                           # preview changes only
  python tools/fix_prompts.py --overwrite                         # overwrite originals
  python tools/fix_prompts.py --system-file my_system.txt          # use custom system prompt from file
  python tools/fix_prompts.py --system-str "You are..." --no-length-check  # inline string, skip token check
  python tools/fix_prompts.py --only-train                         # fix train only
  python tools/fix_prompts.py --only-test                          # fix test only
"""

import argparse, copy, os, sys

import pandas as pd

# ============================================================
# Default improved system prompt (python-only mode: no thinking / <model>)
# ============================================================
DEFAULT_SYSTEM = """You are a helpful Assistant with expertise in mathematical modeling and the Gurobi solver.
Given an optimization problem, provide the Gurobi Python code to solve it.

Your response MUST follow this structure exactly:

<python>
Provide the complete Gurobi Python code to implement the model.
IMPORTANT: Always include `from gurobipy import *` at the top.
Print the optimal objective value and optimal solution clearly.
</python>

Do NOT include any thinking steps or mathematical model sections.
Only output the <python> code block."""

DEFAULT_USER_TEMPLATE = "Solve the following mathematical modeling problem using Gurobi.\n{question}\nProvide only the Gurobi Python code."


def extract_question(user_content: str) -> str:
    """Extract the question body from a user message, dropping common boilerplate."""
    s = user_content.strip()

    # Strip "think step by step" suffix (case-insensitive, with/without newline)
    for suffix in ("\nthink step by step.", "\nthink step by step",
                   "\nThink step by step.", "\nThink step by step",
                   " think step by step.", " Think step by step.",
                   "think step by step.", "Think step by step.",
                   "\nthink step by step", "\nThink step by step"):
        if s.lower().endswith(suffix.lower()):
            s = s[:-len(suffix)].rstrip()
            break

    # Strip known prefixes
    for prefix in (
        "Solve the following mathmetical modeling problem",
        "Solve the following mathematical modeling problem",
    ):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            break

    return s


def build_messages(system_content: str, question: str) -> list:
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": DEFAULT_USER_TEMPLATE.format(question=question)},
    ]


def patch_row(row, system_content: str):
    """Return a modified copy of the row with updated system+user prompts."""
    row = copy.deepcopy(row)
    messages = row.get("prompt", [])

    # ── 兼容 numpy.ndarray ──
    import numpy as np
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if not isinstance(messages, list) or len(messages) == 0:
        return row

    question = ""
    for msg in messages:
        # msg may be a dict-like numpy record; force to dict
        if not isinstance(msg, dict):
            msg = dict(msg)
        if msg.get("role") == "user":
            question = extract_question(msg.get("content", ""))

    if not question:
        return row  # couldn't extract question, skip

    row["prompt"] = build_messages(system_content, question)
    return row


def process_one(label, path, system_content, suffix, overwrite, dry_run, check_len, tokenizer):
    if not os.path.exists(path):
        print(f"[SKIP] {label}: {path} not found\n")
        return

    df = pd.read_parquet(path)
    n = len(df)
    print(f"[READ] {label}: {n} rows from {os.path.basename(path)}")

    # --- Diff preview (first row) ---
    orig = df["prompt"].iloc[0]
    def _get(msgs, role):
        return next((m["content"] for m in msgs if m["role"] == role), "")
    old_sys = _get(orig, "system")
    old_usr = _get(orig, "user")

    df = df.apply(lambda r: patch_row(r, system_content), axis=1)

    new = df["prompt"].iloc[0]
    new_sys = _get(new, "system")
    new_usr = _get(new, "user")

    print(f"  [system OLD] {old_sys[:120]}...")
    print(f"  [system NEW] {new_sys[:120]}...")
    print(f"  [user OLD]   {old_usr[:200]}...")
    print(f"  [user NEW]   {new_usr[:200]}...")

    # --- Token-length check ---
    if check_len and tokenizer is not None:
        max_tok = 0
        for i in range(min(100, n)):
            ids = tokenizer.apply_chat_template(df["prompt"].iloc[i], add_generation_prompt=True, tokenize=True)
            max_tok = max(max_tok, len(ids))
        print(f"  max prompt tokens (first 100 rows): {max_tok}  (limit: 2048)")

    if dry_run:
        print(f"  [DRY-RUN] Would write {label}\n")
        return

    out = path if overwrite else path.replace(".parquet", f"{suffix}.parquet")
    df.to_parquet(out, index=False)
    print(f"  [WRITE] {label} -> {out} ({n} rows)\n")


def main():
    ap = argparse.ArgumentParser(description="Patch system/user prompts in Gurobi OR parquet datasets")
    ap.add_argument("--trainset", default="/home/guo/LLM/Verl/verl/trainset/gurobi_examples_OR_train.parquet")
    ap.add_argument("--testset",  default="/home/guo/LLM/Verl/verl/trainset/gurobi_examples_OR_test.parquet")
    ap.add_argument("--system-file", default=None, help="File containing new system prompt")
    ap.add_argument("--system-str",  default=None, help="New system prompt as inline string")
    ap.add_argument("--suffix", default="_fixed", help="Output file suffix (default: _fixed)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite original files")
    ap.add_argument("--dry-run", action="store_true", help="Show diff without writing")
    ap.add_argument("--no-length-check", action="store_true", help="Skip token-length check")
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--only-test",  action="store_true")
    args = ap.parse_args()

    # Resolve system prompt
    if args.system_file:
        with open(args.system_file) as f:
            system_content = f.read().strip()
    elif args.system_str:
        system_content = args.system_str
    else:
        system_content = DEFAULT_SYSTEM

    # Tokenizer (for length check)
    tokenizer = None
    if not args.no_length_check:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "/home/guo/LLM/SIRL/Qwen3-4B-Instruct-2507", trust_remote_code=True
            )
        except Exception as e:
            print(f"[WARN] tokenizer load failed, skipping length check: {e}")

    targets = []
    if args.only_test:
        targets.append(("test", args.testset))
    elif args.only_train:
        targets.append(("train", args.trainset))
    else:
        targets = [("train", args.trainset), ("test", args.testset)]

    for label, path in targets:
        process_one(label, path, system_content, args.suffix, args.overwrite,
                     args.dry_run, not args.no_length_check, tokenizer)

    if args.dry_run:
        print("Done (dry-run).  Use without --dry-run to write files.")
    else:
        print("Done. Remember to update data.train_files / data.val_files in your run script.")


if __name__ == "__main__":
    main()