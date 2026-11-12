"""Planner backends: a local Hugging Face chat model, and a scripted stand-in for tests and dry runs."""
import json

from agent.loop import LLMReply


class ScriptedLLM:
    """Replies from a fixed list (or a function of the messages); records every call."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, messages):
        self.calls.append(list(messages))
        text = self.replies(messages) if callable(self.replies) else self.replies[min(len(self.calls), len(self.replies)) - 1]
        return LLMReply(text, input_tokens=sum(len(m["content"]) // 4 for m in messages), output_tokens=len(text) // 4)


class ReplayModel:
    """Stand-in planner that repeats what a finished run's planner did, to rebuild its prompts offline (no GPU).

    `load(actions)` takes the `actions` a run logged for one line (one entry per planner reply: the parsed JSON action, or
    None when the reply was not a valid action); each call then answers with the next one. The tools are deterministic, so
    the agent loop goes through the same steps and ends with the same context as in the original run."""

    def __init__(self):
        self.actions, self.position = [], 0

    def load(self, actions):
        self.actions, self.position = list(actions), 0

    def __call__(self, messages):
        if self.position >= len(self.actions):
            raise RuntimeError("replay asked for more planner replies than the run logged")
        action = self.actions[self.position]
        self.position += 1
        return LLMReply(json.dumps(action) if action is not None else "(not a valid action)")

    @property
    def finished(self) -> bool:
        """True when every logged reply was used, as it must be if the replay followed the original run."""
        return self.position == len(self.actions)


class LocalChatModel:
    """An instruction-tuned causal LM run greedily; loaded on first use so importing this module is cheap.

    `dtype` is bfloat16 (what the CodeLlama writer uses), 8bit or 4bit (the quantized ones need bitsandbytes) so the
    planner can sit next to the CodeLlama-7B that writes the line on one 32 GB GPU.
    """

    def __init__(self, name: str, max_new_tokens: int = 200, dtype: str = "bfloat16", device: str = "cuda"):
        if dtype not in ("bfloat16", "8bit", "4bit"):
            raise ValueError(f"dtype must be bfloat16, 8bit or 4bit, got {dtype!r}")
        self.name, self.max_new_tokens, self.dtype, self.device = name, max_new_tokens, dtype, device
        self.model = self.tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.dtype == "bfloat16":
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.bfloat16).to(self.device).eval()
            return
        quantization = BitsAndBytesConfig(load_in_8bit=self.dtype == "8bit", load_in_4bit=self.dtype == "4bit",
                                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, quantization_config=quantization, device_map={"": self.device}).eval()

    def __call__(self, messages):
        import torch

        if self.model is None:
            self._load()
        # render the chat template to text first: its tokenized return type differs between transformers versions
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        return LLMReply(self.tokenizer.decode(new, skip_special_tokens=True), int(ids.shape[1]), int(new.shape[0]))
