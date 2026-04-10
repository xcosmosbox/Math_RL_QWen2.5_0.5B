from __future__ import annotations

import re
from dataclasses import dataclass


FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*final answer\s*[:：]\s*(.+?)\s*$")
GENERIC_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractionResult:
    extracted_answer: str | None
    extraction_method: str
    format_pass: bool
    parse_flags: list[str]


def normalize_answer_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = normalized.strip("$")
    normalized = re.sub(r"^\\boxed\{(.+)\}$", r"\1", normalized)
    normalized = normalized.strip()
    normalized = normalized.rstrip("。．. ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized


def extract_boxed_answer(text: str) -> str | None:
    if "\\boxed{" not in text:
        return None
    start = text.rfind("\\boxed{")
    cursor = start + len("\\boxed{")
    depth = 1
    collected: list[str] = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(collected).strip()
        if depth > 0:
            collected.append(char)
        cursor += 1
    return None


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def has_expected_final_answer_format(text: str) -> bool:
    last_line = last_nonempty_line(text)
    return last_line.lower().startswith("final answer:") or "\\boxed{" in last_line


def _extract_with_regex(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_final_answer(completion: str) -> ExtractionResult:
    text = completion or ""
    flags: list[str] = []

    explicit = _extract_with_regex(FINAL_ANSWER_LINE_RE, text)
    if explicit:
        return ExtractionResult(
            extracted_answer=normalize_answer_text(explicit),
            extraction_method="final_answer_line",
            format_pass=has_expected_final_answer_format(text),
            parse_flags=flags,
        )

    boxed = extract_boxed_answer(text)
    if boxed:
        flags.append("used_boxed_fallback")
        return ExtractionResult(
            extracted_answer=normalize_answer_text(boxed),
            extraction_method="boxed",
            format_pass=has_expected_final_answer_format(text),
            parse_flags=flags,
        )

    generic = _extract_with_regex(GENERIC_ANSWER_LINE_RE, text)
    if generic:
        flags.append("used_generic_answer_fallback")
        return ExtractionResult(
            extracted_answer=normalize_answer_text(generic),
            extraction_method="generic_answer_line",
            format_pass=has_expected_final_answer_format(text),
            parse_flags=flags,
        )

    flags.append("answer_not_found")
    return ExtractionResult(
        extracted_answer=None,
        extraction_method="none",
        format_pass=False,
        parse_flags=flags,
    )


def build_final_answer_suffix(answer: str) -> str:
    return f"Final Answer: {normalize_answer_text(answer)}"
