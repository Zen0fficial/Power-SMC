"""Helpers to extract final boxed answers from model completions."""

import json

import numpy as np


def remove_boxed(s):
    """Strip a leading \\boxed{...} wrapper and return inner content."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def last_boxed_only_string(string):
    """
    Find the last \\boxed{...} or \\fbox{...} with non-empty, non-template content,
    searching right-to-left so we can skip placeholder boxes like \\boxed{{}}.
    """
    search_terms = ["\\boxed", "\\fbox"]

    # Collect all start positions for all search terms
    candidates = []
    for term in search_terms:
        start = 0
        while True:
            idx = string.find(term, start)
            if idx < 0:
                break
            candidates.append(idx)
            start = idx + 1

    if not candidates:
        return None

    # Search from rightmost to leftmost, return first valid non-empty content
    for idx in sorted(candidates, reverse=True):
        i = idx
        # Advance to the opening brace
        while i < len(string) and string[i] != "{":
            i += 1

        if i >= len(string):
            continue

        right_brace_idx = None
        num_left_braces_open = 0
        j = i
        while j < len(string):
            if string[j] == "{":
                num_left_braces_open += 1
            if string[j] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = j
                    break
            j += 1

        if right_brace_idx is None:
            continue

        retval = string[idx : right_brace_idx + 1]
        content = remove_boxed(retval) if retval.startswith("\\boxed{") else retval

        # Skip empty or pure-brace template boxes like \boxed{} or \boxed{{}}
        if (
            content is not None
            and content.strip().replace("{", "").replace("}", "").strip() != ""
        ):
            return retval

    return None


def last_boxed_only(sample):
    """Return (question, last_boxed_answer) for a (question, answer) pair."""
    q, a = sample
    a = last_boxed_only_string(a)
    if a is None:
        return None
    return (q, a)


def parse_answer(input_str):
    """Extract and return the final boxed answer from raw model output text."""
    return remove_boxed(last_boxed_only_string(input_str))
