"""Strict JSON validation and a complete BM25-tail file permutation."""

import json

from bug_localization.rag.prepare_repositories import normalize_path


def validate_response(raw, allowed):
    """Normalize only backslashes to slashes, using step 1's path validator.

    Preserve case and whitespace; reject dot segments, absolute paths, markdown
    wrappers, fuzzy aliases, and unexpected structures. An empty/all-invalid
    list is unusable. Partial usable lists get a deterministic BM25 completion.
    """
    problems, valid = [], []
    try:
        # Duplicate JSON keys are ambiguous, rather than silently last-wins.
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        def reject_constant(value):
            raise ValueError("Non-JSON numeric constant")
        parsed = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return {"status": "fallback_validation", "parsed": None, "ranking": [],
                "problems": [{"type": "invalid_json"}], "repairs": []}
    if not isinstance(parsed, dict) or set(parsed) != {"ranked_files"} or not isinstance(parsed["ranked_files"], list):
        return {"status": "fallback_validation", "parsed": parsed, "ranking": [],
                "problems": [{"type": "invalid_structure"}], "repairs": []}
    repairs, seen, allowed_set = [], set(), set(allowed)
    for position, path in enumerate(parsed["ranked_files"]):
        try:
            if not isinstance(path, str):
                raise ValueError("Non-string path")
            normalized = normalize_path(path)
            if normalized not in allowed_set:
                raise ValueError("Path outside allowed set")
        except ValueError:
            problems.append({"type": "invalid_path", "position": position, "value": path})
            repairs.append({"action": "remove_invalid", "position": position})
            continue
        if normalized in seen:
            problems.append({"type": "duplicate_path", "position": position, "value": path})
            repairs.append({"action": "remove_duplicate", "position": position})
            continue
        if normalized != path:
            problems.append({"type": "path_normalized", "position": position, "value": path})
            repairs.append({"action": "replace_backslashes", "path": normalized})
        seen.add(normalized)
        valid.append(normalized)
    omitted = [path for path in allowed if path not in seen]
    if omitted:
        problems.append({"type": "omitted_candidates", "paths": omitted})
    if not valid:
        return {"status": "fallback_validation", "parsed": parsed, "ranking": [],
                "problems": problems, "repairs": repairs}
    if omitted:
        repairs.append({"action": "append_in_bm25_order", "paths": omitted})
    return {"status": "repaired" if problems else "success", "parsed": parsed,
            "ranking": valid + omitted, "problems": problems, "repairs": repairs}


def complete_ranking(candidate_ranking, bm25_ranking):
    universe = [row["file"] for row in bm25_ranking]
    if (len(universe) != len(set(universe)) or len(candidate_ranking) != len(set(candidate_ranking))
            or not set(candidate_ranking).issubset(universe)):
        raise ValueError("Ranking is not a candidate-universe permutation")
    front = set(candidate_ranking)
    paths = candidate_ranking + [path for path in universe if path not in front]
    # Evaluator requires descending scores. These ordinal scores express the new
    # order only, not BM25 relevance or calibrated model probabilities.
    return [{"rank": i, "file": path, "score": len(paths) - i + 1} for i, path in enumerate(paths, 1)]
