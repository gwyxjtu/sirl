"""Partial-KL penalty: mask KL to tokens from <python tag through response end."""

from __future__ import annotations

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import masked_mean

# Tag lists populated by init_partial_kl_context(tokenizer)
_CODE_TOKEN_LIST: list | None = None
_CODE_TOKEN_LENGTH: list | None = None


def init_partial_kl_context(tokenizer) -> None:
    global _CODE_TOKEN_LIST, _CODE_TOKEN_LENGTH
    _CODE_TOKEN_LIST, _CODE_TOKEN_LENGTH = load_tags("python", tokenizer)


def load_tags(name, tokenizer):
    tag_start_list = [f"<{name}", f" <{name}", f"\n<{name}", f"\n\n<{name}"]
    tag_token_ids = []
    tag_lengths = []
    for tag in tag_start_list:
        ids = tokenizer(tag, add_special_tokens=False)["input_ids"]
        tag_token_ids.append(ids)
        tag_lengths.append(len(ids))
    return tag_token_ids, tag_lengths


def find_tags_tensor(tag_token_ids, tag_lengths, tokenized_texts):
    batch_start_pos = torch.zeros(len(tokenized_texts))

    for b, tokenized_text in enumerate(tokenized_texts):
        for target, tag_length in zip(tag_token_ids, tag_lengths):
            target = torch.tensor(target)
            windows = tokenized_text.unfold(0, tag_length, 1)
            matches = (windows == target).all(dim=1)
            if any(matches):
                batch_start_pos[b] = torch.where(matches)[0][0]
                break
    return batch_start_pos


def partial_kl_tensor(idx_list_start, idx_list_end, response_mask):
    batchsize = idx_list_start.shape[0]
    mask = response_mask.clone()
    for i in range(batchsize):
        idx_start = int(idx_list_start[i].item()) if torch.is_tensor(idx_list_start[i]) else int(idx_list_start[i])
        idx_end = int(idx_list_end[i].item()) if torch.is_tensor(idx_list_end[i]) else int(idx_list_end[i])
        if idx_end <= idx_start:
            idx_end = mask.shape[1]
        mask[i, :idx_start] = 0
        mask[i, idx_end:] = 0
    return mask


def apply_partial_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    """Apply Partial-KL penalty: only <python segment contributes to KL in reward."""
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    if _CODE_TOKEN_LIST is not None:
        code_idx = find_tags_tensor(_CODE_TOKEN_LIST, _CODE_TOKEN_LENGTH, data.batch["responses"])
        code_kl_mask = partial_kl_tensor(code_idx, torch.zeros_like(code_idx), response_mask)
        kld = kld * code_kl_mask
        masked_kld = masked_mean(kld, mask=code_kl_mask, axis=-1)
        masked_kld = torch.mean(masked_kld, dim=0).item()
    else:
        masked_kld = current_kl

    token_level_rewards = token_level_scores - beta * kld
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_partial_kl_penalty": masked_kld,
        "actor/reward_kl_penalty_coeff": beta,
    }
    return data, metrics
