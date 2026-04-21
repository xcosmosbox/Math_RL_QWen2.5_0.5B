from __future__ import annotations

import re
from dataclasses import dataclass


ANSWER_TAG_RE = re.compile(r"(?is)<answer>\s*(.+?)\s*</answer>")
ANSWER_TAG_SPAN_RE = re.compile(r"(?is)<answer>\s*.+?\s*</answer>")
FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*final answer\s*[:：]\s*(.+?)\s*$")
GENERIC_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$")
WHITESPACE_RE = re.compile(r"\s+")
DISPLAY_CLOSER_RE = re.compile(r"^\s*(?:\\\]|\\\)|\$\$|[.。,，:：;；!?！？]*)\s*$")
UNICODE_SQRT_RE = re.compile(r"√\s*(\([^()]+\)|\{[^{}]+\}|[A-Za-z0-9]+)")
ROLE_MARKER_RE = re.compile(r"(?i)(?:<\|im_start\|>\s*(?:user|assistant)|\b(?:human|assistant|user)\b\s*:?)")


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
    answer_tag_match = ANSWER_TAG_RE.fullmatch(normalized)
    if answer_tag_match:
        normalized = answer_tag_match.group(1).strip()
    normalized = normalized.strip("$")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("—", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("∕", "/")
    normalized = normalized.replace("÷", "/")
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = UNICODE_SQRT_RE.sub(_replace_unicode_sqrt, normalized)
    boxed = extract_last_boxed_span(normalized)
    if boxed is not None:
        boxed_text = boxed[0].strip()
        remainder = (normalized[: boxed[1]] + normalized[boxed[2] :]).strip()
        if not remainder or len(remainder) <= 24:
            normalized = boxed_text
    normalized = re.sub(r"^\\boxed\{(.+)\}$", r"\1", normalized)
    normalized = normalized.strip()
    normalized = normalized.rstrip("。．. ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized


def _replace_unicode_sqrt(match: re.Match[str]) -> str:
    body = match.group(1).strip()
    if (body.startswith("(") and body.endswith(")")) or (body.startswith("{") and body.endswith("}")):
        body = body[1:-1].strip()
    return rf"\sqrt{{{body}}}"


def extract_boxed_answer(text: str) -> str | None:
    boxed = extract_last_boxed_span(text)
    if boxed is None:
        return None
    return boxed[0]


def extract_last_boxed_span(text: str) -> tuple[str, int, int] | None:
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
                return ("".join(collected).strip(), start, cursor + 1)
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
    return bool(last_line.lower().startswith("final answer:"))


def has_deepmath_final_answer_format(text: str) -> bool:
    if has_expected_final_answer_format(text):
        return True
    boxed = extract_last_boxed_span(text)
    if boxed is None:
        return False
    _, _, end = boxed
    tail = text[end:].strip()
    return bool(DISPLAY_CLOSER_RE.fullmatch(tail))


def _extract_with_regex(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def trim_completion_to_answer(completion: str) -> tuple[str, list[str]]:
    text = completion or ""
    flags: list[str] = []

    answer_tag_matches = list(ANSWER_TAG_SPAN_RE.finditer(text))
    if answer_tag_matches:
        end = answer_tag_matches[-1].end()
        trimmed = text[:end].rstrip()
        if trimmed != text.rstrip():
            flags.append("trimmed_after_answer_anchor")
        return trimmed, flags

    explicit_matches = list(FINAL_ANSWER_LINE_RE.finditer(text))
    if explicit_matches:
        end = explicit_matches[-1].end()
        trimmed = _truncate_at_role_marker(text[:end].rstrip(), flags)
        if trimmed != text.rstrip():
            flags.append("trimmed_after_answer_anchor")
        return trimmed, flags

    boxed = extract_last_boxed_span(text)
    if boxed is not None:
        _, _, end = boxed
        trimmed = _truncate_at_role_marker(text[:end].rstrip(), flags)
        if trimmed != text.rstrip():
            flags.append("trimmed_after_answer_anchor")
        return trimmed, flags

    return text, flags


def _truncate_at_role_marker(text: str, flags: list[str]) -> str:
    match = ROLE_MARKER_RE.search(text)
    if match is None:
        return text
    flags.append("trimmed_at_role_marker")
    return text[: match.start()].rstrip()


def extract_final_answer(completion: str, *, format_profile: str = "default") -> ExtractionResult:
    text, flags = trim_completion_to_answer(completion or "")
    format_check = has_expected_final_answer_format
    if format_profile == "deepmath":
        format_check = has_deepmath_final_answer_format

    answer_tag = _extract_with_regex(ANSWER_TAG_RE, text)
    if answer_tag:
        return ExtractionResult(
            extracted_answer=normalize_answer_text(answer_tag),
            extraction_method="answer_tag",
            format_pass=format_check(text),
            parse_flags=flags,
        )

    explicit = _extract_with_regex(FINAL_ANSWER_LINE_RE, text)
    if explicit:
        return ExtractionResult(
            extracted_answer=normalize_answer_text(explicit),
            extraction_method="final_answer_line",
            format_pass=format_check(text),
            parse_flags=flags,
        )

    boxed = extract_boxed_answer(text)
    if boxed:
        flags.append("used_boxed_fallback")
        return ExtractionResult(
            extracted_answer=normalize_answer_text(boxed),
            extraction_method="boxed",
            format_pass=format_check(text),
            parse_flags=flags,
        )

    generic = _extract_with_regex(GENERIC_ANSWER_LINE_RE, text)
    if generic:
        flags.append("used_generic_answer_fallback")
        return ExtractionResult(
            extracted_answer=normalize_answer_text(generic),
            extraction_method="generic_answer_line",
            format_pass=format_check(text),
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
