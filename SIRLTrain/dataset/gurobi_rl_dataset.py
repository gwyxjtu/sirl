"""RLHF dataset for Gurobi OR: drop columns that conflict with VeRL 0.8 AgentLoop kwargs."""

from verl.utils.dataset.rl_dataset import RLHFDataset

# VeRL AgentLoop._agent_loop_postprocess(..., output, **kwargs) conflicts with parquet "output".
# Other raw fields are unused after __getitem__ builds raw_prompt / reward fields.
_DROP_KEYS = frozenset(
    {
        "output",
        "en_question",
        "en_answer",
        "sol",
        "prompt",
    }
)


class GurobiRLHFDataset(RLHFDataset):
    def _build_messages(self, example: dict, key: str):
        messages = example[key]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if isinstance(messages, dict):
            messages = [messages]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return [dict(message) for message in messages]

    def __getitem__(self, item):
        row = super().__getitem__(item)
        for key in _DROP_KEYS:
            row.pop(key, None)
        return row
