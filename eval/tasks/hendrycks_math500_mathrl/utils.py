from __future__ import annotations

import re
from typing import Dict, List

import datasets


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
TEXT_ENV_RE = re.compile(r"\\text\{([^{}]*)\}")
FINAL_PATTERNS = [
    re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"The final answer is\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"The answer is\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"(?:Therefore|Thus|So),?\s*(?:the\s+)?(?:answer|value|result|student|coordinates?|solution)[^.\n]*?is\s*(.*)",
        re.IGNORECASE | re.DOTALL,
    ),
]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    target = remove_boxed(last_boxed_only_string(doc["solution"]))
    candidates = extract_model_answers(results[0])
    retval = 1 if any(is_equiv(candidate, target) for candidate in candidates) else 0
    return {"exact_match": retval}


def clean_candidate(text: str) -> str:
    text = (text or "").strip()
    text = text.strip("`")
    for left, right in (("\\(", "\\)"), ("\\[", "\\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"^(?:The final answer is|The answer is|Final Answer:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(".。 ")
    return text


def extract_model_answers(text: str) -> list[str]:
    candidates: list[str] = []
    text = text or ""

    match = ANSWER_TAG_RE.search(text or "")
    if match:
        candidates.append(match.group(1).strip())

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            candidates.append(remove_boxed(boxed))
        except Exception:
            pass

    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            candidates.append(match.group(1).strip())

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(non_empty_lines[-8:]):
        if line in {"\\[", "\\]", "$"}:
            continue
        candidates.append(line)
        if "=" in line:
            candidates.append(line.split("=")[-1].strip())
        for math_match in INLINE_MATH_RE.findall(line):
            candidates.append(math_match.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = clean_candidate(candidate)
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped or [clean_candidate(text)]


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("<answer>", "").replace("</answer>", "")
    string = TEXT_ENV_RE.sub(r"\1", string)
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string
