"""Planner backends: a local Hugging Face chat model, and a scripted stand-in for tests and dry runs."""
from agent.loop import LLMReply


class ScriptedLLM:
    """Replies from a fixed list (or a function of the messages); records every call."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, messages):
        self.calls.append(list(messages))
        text = self.replies(messages) if callable(self.replies) else self.replies[min(len(self.calls), len(self.replies)) - 1]
        return LLMReply(text, input_tokens=sum(len(m["content"]) // 4 for m in messages), output_tokens=len(text) // 4)


class LocalChatModel:
    """An instruction-tuned causal LM run greedily; loaded on first use so importing this module is cheap.

    `dtype` is float16, 8bit or 4bit (the quantized ones need bitsandbytes) so the planner can sit next to the
    CodeLlama-7B that writes the line on one 32 GB GPU.
    """

    def __init__(self, name: str, max_new_tokens: int = 200, dtype: str = "float16", device: str = "cuda"):
        if dtype not in ("float16", "8bit", "4bit"):
            raise ValueError(f"dtype must be float16, 8bit or 4bit, got {dtype!r}")
        self.name, self.max_new_tokens, self.dtype, self.device = name, max_new_tokens, dtype, device
        self.model = self.tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.dtype == "float16":
            self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device).eval()
            return
        quantization = BitsAndBytesConfig(load_in_8bit=self.dtype == "8bit", load_in_4bit=self.dtype == "4bit",
                                          bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, quantization_config=quantization, device_map={"": self.device}).eval()

    def __call__(self, messages):
        import torch

        if self.model is None:
            self._load()
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        return LLMReply(self.tokenizer.decode(new, skip_special_tokens=True), int(ids.shape[1]), int(new.shape[0]))
