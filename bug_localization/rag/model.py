"""Small OpenAI chat interface; no dataset access and no validation retry loop."""

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol


RANKING_SCHEMA = {
    "type": "object", "properties": {
        "ranked_files": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranked_files"], "additionalProperties": False,
}


@dataclass(frozen=True)
class ModelSettings:
    backend: str = "openai"
    model_name: str = "gpt-4"
    context_window: int = 8192
    max_output_tokens: int = 2048
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    response_format: str = "text"
    tokenizer: str = "auto"
    safety_margin: int = 256

    def validate(self):
        if self.backend != "openai" or not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Only the openai backend is supported; require a model_name")
        for name in ("context_window", "max_output_tokens", "max_retries", "safety_margin"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ("context_window", "max_output_tokens") else 0):
                raise ValueError(f"Invalid {name}")
        if self.max_output_tokens + self.reserved_safety_tokens >= self.context_window:
            raise ValueError("Output and safety reservation exhaust the context window")
        for name, upper in (("temperature", 2), ("top_p", 1)):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= upper):
                raise ValueError(f"Invalid {name}")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or null")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.response_format not in ("json_schema", "json_object", "text"):
            raise ValueError("Unknown response_format")
        if self.tokenizer not in ("auto", "bytes"):
            raise ValueError("tokenizer must be auto or bytes")

    @property
    def reserved_safety_tokens(self):
        return max(self.safety_margin, math.ceil(self.context_window * .05))

    @property
    def input_budget(self):
        return self.context_window - self.max_output_tokens - self.reserved_safety_tokens

    def format_parameter(self):
        if self.response_format == "json_schema":
            return {"type": "json_schema", "json_schema": {
                "name": "file_ranking", "strict": True, "schema": RANKING_SCHEMA}}
        return {"type": self.response_format}

    def request_parameters(self):
        token_parameter = "max_tokens" if self.model_name in ("gpt-4", "gpt-4-0613", "gpt-4-0314") else "max_completion_tokens"
        parameters = {"model": self.model_name, token_parameter: self.max_output_tokens,
                      "n": 1, "stream": False}
        if self.response_format != "text":
            parameters["response_format"] = self.format_parameter()
        for name in ("temperature", "top_p", "seed"):
            if getattr(self, name) is not None:
                parameters[name] = getattr(self, name)
        return parameters

    def identity(self):
        # No arbitrary kwargs, endpoints, headers, or credentials can enter this record.
        return {**asdict(self), "request_parameters": self.request_parameters(),
                "endpoint": "https://api.openai.com/v1", "sdk_max_retries": 0}


@dataclass
class ModelResponse:
    raw_response: str | None
    usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    attempts: list = field(default_factory=list)
    request_id: str | None = None
    response_id: str | None = None
    returned_model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    refusal: str | None = None
    error: dict | None = None


class ModelClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> ModelResponse: ...


class OpenAIClient:
    def __init__(self, settings):
        settings.validate()
        self.settings = settings

    def generate(self, messages):
        started = time.perf_counter()
        attempts = []
        try:
            import openai
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                   base_url="https://api.openai.com/v1",
                                   timeout=self.settings.timeout_seconds, max_retries=0)
        except (ImportError, RuntimeError) as error:
            return ModelResponse(None, latency_seconds=time.perf_counter() - started,
                                 error={"type": type(error).__name__, "stage": "client_setup"})
        try:
            for attempt in range(self.settings.max_retries + 1):
                attempt_start = time.perf_counter()
                try:
                    completion = client.chat.completions.create(
                        messages=messages, **self.settings.request_parameters())
                except openai.APIError as error:
                    status = getattr(error, "status_code", None)
                    retryable = (isinstance(error, openai.APIConnectionError)
                                 or status in (408, 409, 429) or (status is not None and status >= 500))
                    # Never serialize exception text/bodies/headers: they can contain credentials.
                    problem = {"type": type(error).__name__, "status_code": status,
                               "request_id": getattr(error, "request_id", None)}
                    attempts.append({"attempt": attempt + 1, "error": problem,
                                     "latency_seconds": time.perf_counter() - attempt_start})
                    if retryable and attempt < self.settings.max_retries:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    return ModelResponse(None, attempts=attempts, error=problem,
                                         latency_seconds=time.perf_counter() - started)
                attempts.append({"attempt": attempt + 1, "status": "response",
                                 "latency_seconds": time.perf_counter() - attempt_start})
                choice = completion.choices[0] if completion.choices else None
                return ModelResponse(
                    raw_response=choice.message.content if choice else None,
                    usage=completion.usage.model_dump(mode="json") if completion.usage else {},
                    latency_seconds=time.perf_counter() - started, attempts=attempts,
                    request_id=getattr(completion, "_request_id", None), response_id=completion.id,
                    returned_model=completion.model,
                    system_fingerprint=getattr(completion, "system_fingerprint", None),
                    finish_reason=choice.finish_reason if choice else None,
                    refusal=getattr(choice.message, "refusal", None) if choice else None)
        finally:
            client.close()
