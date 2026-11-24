"""Deterministic API-context packing without reference data or model calls."""

PROMPT = ("Generate Python code based on the following instruction. Output ONLY code. "
          "DO NOT include explanations or other textual content.\nInstruction: ")
HINT = "\n\nYou can find the following APIs from the library helpful:\n"


def pack_context(instruction, apis, mode, budget, count, context_window=128000,
                 max_output_tokens=2048, safety_tokens=1024, deduplicate=True,
                 ordered_apis=None):
    if mode not in ("plain", "direct", "rag"):
        raise ValueError("Unknown context mode")
    for value in (budget, context_window, max_output_tokens, safety_tokens):
        if type(value) is not int or value < 0:
            raise ValueError("Token settings must be nonnegative integers")
    if context_window <= max_output_tokens + safety_tokens:
        raise ValueError("Context window must leave room for input")
    if mode == "plain" and budget:
        raise ValueError("Plain mode requires a zero context budget")
    if not isinstance(instruction, str) or any(not isinstance(a, str) for a in apis):
        raise ValueError("Instruction and API names must be strings")
    candidates = list(dict.fromkeys(apis)) if deduplicate else list(apis)
    ordered = candidates if ordered_apis is None else list(ordered_apis)
    if mode == "rag" and candidates and ordered_apis is None:
        from .rag_retriever import RAGRetriever
        ordered = RAGRetriever(candidates, mode="bm25").top_k(instruction, len(candidates))

    def render(names):
        return PROMPT + instruction + (HINT + ", ".join(names) if names else "")

    # Conservative framing estimate for the single user message.
    input_budget = context_window - max_output_tokens - safety_tokens
    if count(render([])) + 32 > input_budget:
        raise ValueError("Instruction alone exceeds input budget; it is not truncated")
    selected, decisions = [], []
    # The common full-fit case requires just one complete request count.
    if mode != "plain" and ordered and count(HINT + ", ".join(ordered)) <= budget and count(render(ordered)) + 32 <= input_budget:
        selected = list(ordered)
        ordered = []
        decisions = [{"api": api, "reason": "included"} for api in selected]
    for api in ordered if mode != "plain" else []:
        trial = selected + [api]
        evidence_tokens = count(HINT + ", ".join(trial))
        if evidence_tokens > budget:
            reason = "evidence_budget"
        elif count(render(trial)) + 32 > input_budget:
            reason = "context_window"
        else:
            reason = "included"
            selected.append(api)
        decisions.append({"api": api, "reason": reason})
    prompt = render(selected)
    return {
        "prompt": prompt, "mode": mode, "requested_context_tokens": budget,
        "actual_context_tokens": count(HINT + ", ".join(selected)) if selected else 0,
        "estimated_input_tokens": count(prompt) + 32, "input_budget": input_budget,
        "candidate_count": len(candidates), "included_count": len(selected),
        "selected_apis": selected, "decisions": decisions,
        "truncated": mode != "plain" and len(selected) < len(candidates),
    }
