from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction

from rewards.answer_extraction import normalize_answer_text


def canonicalize_verifier_text(text: str | None) -> str:
    normalized = normalize_answer_text(text)
    if len(normalized) == 1 and normalized.isalpha():
        return normalized.upper()
    lowered = normalized.lower()
    if lowered in {"yes", "no", "true", "false"}:
        return lowered
    return normalized


def exact_match(predicted: str | None, target: str | None) -> bool:
    return canonicalize_verifier_text(predicted) == canonicalize_verifier_text(target)


def _parse_number(text: str) -> Decimal | None:
    normalized = normalize_answer_text(text)
    if not normalized:
        return None
    try:
        if "/" in normalized and normalized.count("/") == 1:
            fraction = Fraction(normalized)
            return Decimal(fraction.numerator) / Decimal(fraction.denominator)
        return Decimal(normalized)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def equivalent_match_offline(predicted: str | None, target: str | None) -> bool:
    if exact_match(predicted, target):
        return True
    if predicted is None or target is None:
        return False
    predicted_num = _parse_number(predicted)
    target_num = _parse_number(target)
    if predicted_num is not None and target_num is not None:
        return predicted_num == target_num
    return False
