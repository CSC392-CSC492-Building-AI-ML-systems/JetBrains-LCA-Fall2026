import os

import numpy as np
from rank_bm25 import BM25Okapi

from .example_generation_model import ExampleGenerationModel
from .utils import split_identifier


class LBCGBaseline(ExampleGenerationModel):
    """Own implementation of the two Library-based Code Generation baselines.

    use_context=False -> plain instruction
    use_context=True  -> instruction + top-k API names ranked by BM25
    Prompt text and tie-breaking deliberately mirror the official code so the
    two implementations can be checked against each other.
    """

    def __init__(self, model_name: str, use_context: bool = False, k: int = 20):
        self.model_name = model_name
        self.use_context = use_context
        self.k = k
        self._client = None  # created lazily so prompt building can be tested offline

    # ---- retrieval -------------------------------------------------------
    def retrieve(self, instruction: str, project_apis: list[str]) -> list[str]:
        bm25 = BM25Okapi([split_identifier(name) for name in project_apis])
        query = "".join(c for c in instruction if c.isalnum() or c == " ").lower().split(" ")
        scores = bm25.get_scores(query)
        order = list(reversed(np.argsort(scores)))[: self.k]
        return [project_apis[i] for i in order]

    # ---- prompt ----------------------------------------------------------
    def build_prompt(self, instruction: str, project_apis: list[str] | None = None) -> str:
        if self.use_context:
            apis = self.retrieve(instruction, project_apis)
            instruction = (
                instruction
                + "\n\nYou can find the following APIs from the library helpful:\n"
                + ", ".join(apis)
            )
        return (
            "Generate Python code based on the following instruction. "
            "Output ONLY code. DO NOT include explanations or other textual content.\n"
            f"Instruction: {instruction}"
        )

    # ---- generation ------------------------------------------------------
    def generate(self, task_description: str, project_apis: list[str] = None) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.build_prompt(task_description, project_apis)}],
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content

    def name(self):
        return f"own_{'bm25' if self.use_context else 'plain'}/{self.model_name}"
