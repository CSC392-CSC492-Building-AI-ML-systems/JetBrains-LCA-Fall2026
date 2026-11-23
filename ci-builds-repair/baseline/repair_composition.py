"""Conservative block composition from old source and model replacements only.

This is a separately evaluated repair policy. It never reads reference repair
text, CI verdicts, retrieval scores, or dataset identifiers.
"""

import ast
import re


def blank_edges(text):
    lines = text.splitlines(keepends=True)
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[:start], lines[start:end], lines[end:]


def preserve_boundaries(old, replacement):
    """Keep source boundary blank lines; preserve all substantive model text.

    Empty/all-whitespace replacements are left intact so explicit deletions
    are not turned into invented code. This policy can prevent a deliberate
    boundary-whitespace edit; its results must remain a separate variant.
    """
    prefix, core, suffix = blank_edges(replacement)
    if not core:
        return replacement
    before, _, after = blank_edges(old)
    # Blank-only old blocks do not provide a meaningful layout anchor.
    if not old.strip():
        return replacement
    # Retain the original final-newline convention, not arbitrary rstrip() of
    # code lines (which could alter whitespace inside multiline strings).
    if not after:
        final = core[-1]
        if old.endswith('\r\n'):
            core[-1] = final.removesuffix('\n').removesuffix('\r') + '\r\n'
        elif old.endswith('\n'):
            core[-1] = final.removesuffix('\n').removesuffix('\r') + '\n'
        elif old.endswith('\r'):
            core[-1] = final.removesuffix('\n').removesuffix('\r') + '\r'
        else:
            core[-1] = final.removesuffix('\n').removesuffix('\r')
    return ''.join([*before, *core, *after])


def compose_replacements(blocks, replacements):
    if len(blocks) != len(replacements):
        raise ValueError('Replacement count mismatch')
    if any(set(block) != {'path', 'start', 'count', 'old'} for block in blocks):
        raise ValueError('Composition accepts old-side oracle data only')
    return [preserve_boundaries(b['old'], text) for b, text in zip(blocks, replacements)]


def compose_files(files, blocks, replacements):
    """Guard against blank lines inside strings and invalid model code.

    Only Python files with identical before/after ASTs use the policy. No
    modification is accepted merely because it improves a CI verdict.
    """
    from verify_completed_run import apply_edits
    original = apply_edits(files, blocks, replacements)
    candidate = apply_edits(files, blocks, compose_replacements(blocks, replacements))
    output, notes = dict(original), {}
    for path, text in candidate.items():
        if text == original[path]:
            notes[path] = 'unchanged'
            continue
        if not path.endswith(('.py', '.pyi')):
            notes[path] = 'non_python_not_modified'
            continue
        try:
            before = ast.dump(ast.parse(original[path]))
            after = ast.dump(ast.parse(text))
        except (SyntaxError, ValueError):
            notes[path] = 'invalid_python_not_modified'
            continue
        if before != after:
            notes[path] = 'ast_change_rejected'
            continue
        output[path] = text
        notes[path] = 'boundary_whitespace_preserved'
    return output, notes


def recover_markers(response, blocks):
    """Recover only complete, unambiguous same-file block-ID envelopes.

    The oracle, not a generated line number, determines where text is applied.
    Accept B/E end-marker aliases; reject unknown paths/IDs, missing/duplicate
    blocks, nesting, and any prose outside envelopes. Never choose among fixes.
    """
    if any(set(b) != {'path', 'start', 'count', 'old'} for b in blocks):
        raise ValueError('Recovery accepts old-side data only')
    text = response.strip()
    fence = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', text, re.S)
    if fence:
        text = fence[1]
    marker = re.compile(r'^\[(start|end) of ([^\n]+?)#L\d+#([BE])(\d+)\][ \t]*$', re.M)
    matches = list(marker.finditer(text))
    if len(matches) != 2 * len(blocks) or not blocks:
        raise ValueError('Incomplete or ambiguous block envelope')
    output, cursor = {}, 0
    for start, end in zip(matches[::2], matches[1::2]):
        index = int(start[4])
        if (text[cursor:start.start()].strip() or start[1] != 'start' or start[3] != 'B'
                or end[1] != 'end' or index != int(end[4]) or index >= len(blocks)
                or start[2] != end[2] or start[2] != blocks[index]['path'] or index in output):
            raise ValueError('Unknown, repeated, or mismatched block envelope')
        body = text[start.end():end.start()]
        if not body.startswith('\n') or not body.endswith('\n'):
            raise ValueError('Markers must delimit complete lines')
        output[index] = body[1:]
        cursor = end.end()
    if text[cursor:].strip() or set(output) != set(range(len(blocks))):
        raise ValueError('Extraneous or missing block content')
    return [output[i] for i in range(len(blocks))]
