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

# ============================================================
# Enhanced system prompt with a worked Gurobi few-shot example.
# Use --with-example flag to select this prompt.
# Designed to demonstrate: binary vars, Big-M, conditional
# penalty, correct API (ObjVal/.X/vtype), demand/capacity.
# ============================================================
DEFAULT_SYSTEM_WITH_EXAMPLE = """You are a helpful Assistant with expertise in mathematical modeling and the Gurobi solver.
Given an optimization problem, provide the Gurobi Python code to solve it.

Your response MUST follow this structure exactly:

<python>
Provide the complete Gurobi Python code to implement the model.
IMPORTANT: Always include `from gurobipy import *` at the top.
Print the optimal objective value and optimal solution clearly.
</python>

--- EXAMPLE (study this before writing your code) ---

Problem: A factory must decide which of 2 machines (M1, M2) to operate to produce 2 products (A, B).
- Demand: A=100, B=150 units
- Machine capacities: M1=200, M2=180 hours
- Production time per unit: M1→A:2h, M1→B:3h, M2→A:4h, M2→B:2h
- Operating cost per machine: M1=$500, M2=$400 (only when used)
- If total production < 200 units, pay a one-time penalty of $1000. Otherwise no penalty.

<python>
from gurobipy import *

# Data
demand  = {'A': 100, 'B': 150}
cap     = {'M1': 200, 'M2': 180}
hours   = {('M1','A'):2, ('M1','B'):3, ('M2','A'):4, ('M2','B'):2}
op_cost = {'M1': 500, 'M2': 400}
penalty = 1000
penalty_threshold = 200

model = Model("Factory")

machines, products = ['M1','M2'], ['A','B']

# --- Variables ---
x = model.addVars(machines, products, lb=0, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(machines, vtype=GRB.BINARY, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")   # 1 = no penalty

# --- Objective ---
total = sum(x[m,p] for m in machines for p in products)
model.setObjective(
    sum(op_cost[m] * y[m] for m in machines) + penalty * (1 - z),
    GRB.MINIMIZE
)

# --- Constraints ---
# 1. Meet demand
for p in products:
    model.addConstr(sum(x[m,p] for m in machines) == demand[p], f"Demand_{p}")

# 2. Capacity (Big-M: only produce if machine is selected)
for m in machines:
    model.addConstr(sum(hours[m,p] * x[m,p] for p in products)
                    <= cap[m] * y[m], f"Capacity_{m}")

# 3. Penalty indicator (z=1 when total >= threshold, else z=0)
model.addConstr(total >= penalty_threshold * z, "Penalty_Indicator")

# --- Solve ---
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal Objective Value:", model.ObjVal)
    for m in machines:
        if y[m].X > 0.5:
            print(f"{m} selected")
            for p in products:
                if x[m,p].X > 0:
                    print(f"  {p}: {x[m,p].X}")
else:
    print("No optimal solution, status:", model.status)
</python>

--- KEY GUROBI RULES (must follow) ---
1. Use model.ObjVal (capital O,V) — NOT objVal; use var.X (capital X) — NOT var.x.
2. Binary: vtype=GRB.BINARY; Integer: vtype=GRB.INTEGER; Continuous: vtype=GRB.CONTINUOUS.
3. Conditional penalties → binary indicator var + Big-M (see z above).  Do NOT use addConstr(>= threshold) + addConstr(<= threshold).
4. Flow variables must be non-negative (lb=0).  lb=-GRB.INFINITY is wrong.
5. Always check model.status == GRB.OPTIMAL before printing results.
---

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

    # Strip "Provide only the Gurobi Python code." suffix (from fixed prompts)
    for suffix in ("\nProvide only the Gurobi Python code.",
                   "\nProvide only the Gurobi Python code",
                   "Provide only the Gurobi Python code.",
                   "Provide only the Gurobi Python code"):
        if s.endswith(suffix):
            s = s[:-len(suffix)].rstrip()
            break

    # Strip known prefixes
    for prefix in (
        "Solve the following mathmetical modeling problem using Gurobi.",
        "Solve the following mathematical modeling problem using Gurobi.",
        "Solve the following mathmetical modeling problem",
        "Solve the following mathematical modeling problem",
    ):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            break

    # Strip leading "using Gurobi." (leftover from prefix match)
    if s.startswith("using Gurobi."):
        s = s[len("using Gurobi."):].strip()
    elif s.startswith("using Gurobi"):
        s = s[len("using Gurobi"):].strip()

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
        print(f"  max prompt tokens (first 100 rows): {max_tok}  (limit: 4096)")

    if dry_run:
        print(f"  [DRY-RUN] Would write {label}\n")
        return

    out = path if overwrite else path.replace(".parquet", f"{suffix}.parquet")
    df.to_parquet(out, index=False)
    print(f"  [WRITE] {label} -> {out} ({n} rows)\n")


def main():
    ap = argparse.ArgumentParser(description="Patch system/user prompts in Gurobi OR parquet datasets")
    ap.add_argument("--trainset", default="/root/llm/sirl/SIRLTrain/trainset/gurobi_examples_OR_train.parquet")
    ap.add_argument("--testset",  default="/root/llm/sirl/SIRLTrain/trainset/gurobi_examples_OR_test.parquet")
    ap.add_argument("--system-file", default=None, help="File containing new system prompt")
    ap.add_argument("--system-str",  default=None, help="New system prompt as inline string")
    ap.add_argument("--with-example", action="store_true", help="Use DEFAULT_SYSTEM_WITH_EXAMPLE (few-shot)")
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
    elif args.with_example:
        system_content = DEFAULT_SYSTEM_WITH_EXAMPLE
    else:
        system_content = DEFAULT_SYSTEM

    # Tokenizer (for length check)
    tokenizer = None
    if not args.no_length_check:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "/root/autodl-tmp/models/Qwen3-8B", trust_remote_code=True
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