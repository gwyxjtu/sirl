#!/usr/bin/env python3
"""
Convert VeRL FSDP checkpoint (6-rank sharded) to standard HuggingFace format
and upload to HuggingFace Hub.

Usage:
    HF_TOKEN="hf_..." python3 scripts/convert_and_upload.py \
        --ckpt_dir /root/autodl-tmp/checkpoints/SIRL/.../global_step_62/actor \
        --output_dir /root/autodl-tmp/ckpt_hf \
        --repo_id theGuo/SIRL-Qwen3-8B
"""

import argparse
import gc
import os
import shutil
import sys
import time

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate


def convert_and_save(ckpt_dir: str, src_hf_dir: str, output_dir: str):
    """Load 6 FSDP shards, reconstruct full tensors, save as safetensors."""
    world_size = 6

    shard_paths = []
    for rank in range(world_size):
        p = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shard: {p}")
        shard_paths.append(p)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {world_size} shards ...")
    raw_shards = []
    for rank, p in enumerate(shard_paths):
        t0 = time.time()
        shard = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        raw_shards.append(shard)
        print(f"  rank {rank}: {len(shard)} keys ({time.time() - t0:.1f}s)")

    keys = list(raw_shards[0].keys())
    for rank in range(1, world_size):
        if set(raw_shards[rank].keys()) != set(keys):
            raise RuntimeError(f"Key mismatch between rank 0 and rank {rank}")
    print(f"Key check OK: {len(keys)} keys consistent across {world_size} ranks")

    print("Reconstructing full tensors from shards ...")
    state_dict = {}
    n_sharded = 0
    n_replicated = 0

    for i, key in enumerate(keys):
        tensors = [raw_shards[r][key] for r in range(world_size)]
        t0 = tensors[0]

        if hasattr(t0, "to_local"):
            placements = t0.placements
            local_tensors = [t.to_local() for t in tensors]

            shard_dim = None
            for d, p in enumerate(placements):
                if isinstance(p, Shard):
                    shard_dim = d
                    break

            if shard_dim is not None:
                full = torch.cat(local_tensors, dim=shard_dim)
                n_sharded += 1
            else:
                full = local_tensors[0].clone()
                n_replicated += 1

            state_dict[key] = full.contiguous()
        else:
            state_dict[key] = t0.contiguous()

        if (i + 1) % 50 == 0 or i == 0:
            size_mb = state_dict[key].numel() * state_dict[key].element_size() / 1e6
            print(f"  [{i+1}/{len(keys)}] {key}  ({size_mb:.1f} MB)  "
                  f"sharded={n_sharded} replicated={n_replicated}")

    print(f"Reconstruction done: {n_sharded} sharded + {n_replicated} replicated")

    for s in raw_shards:
        del s
    del raw_shards
    gc.collect()

    print("Saving model.safetensors ...")
    t0 = time.time()
    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"  Saved in {time.time() - t0:.1f}s")

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

    del state_dict
    gc.collect()

    print("Copying config & tokenizer files ...")
    for fname in os.listdir(src_hf_dir):
        src = os.path.join(src_hf_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print("  Done.")


def upload_to_hf(local_dir: str, repo_id: str, private: bool = False):
    """Upload to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Run 'export HF_TOKEN=hf_...' first.")
        sys.exit(1)

    api = HfApi(token=token)

    try:
        who = api.whoami()
        print(f"Logged in as: {who['name']}")
    except Exception as e:
        print(f"ERROR: Login failed: {e}")
        sys.exit(1)

    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repo: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"WARNING: Could not create/access {repo_id}: {e}")

    print(f"Uploading files to {repo_id} ...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done! https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert VeRL FSDP ckpt -> HF format")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--output_dir", default="/root/autodl-tmp/ckpt_hf")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    src_hf_dir = os.path.join(args.ckpt_dir, "huggingface")
    if not os.path.isdir(src_hf_dir):
        print(f"ERROR: huggingface/ dir not found at {src_hf_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Step 1/2: Converting FSDP model to HuggingFace format ...")
    convert_and_save(args.ckpt_dir, src_hf_dir, args.output_dir)

    total_bytes = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if os.path.isfile(os.path.join(args.output_dir, f))
    )
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")

    if args.repo_id:
        print(f"\nStep 2/2: Uploading to {args.repo_id} ...")
        upload_to_hf(args.output_dir, args.repo_id, private=args.private)
    else:
        print(f"\nModel saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
