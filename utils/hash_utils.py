from __future__ import annotations

import hashlib
import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text_for_hash(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return WHITESPACE_RE.sub(" ", text)


def stable_hash(text: str, *, digest_size: int = 12) -> str:
    normalized = normalize_text_for_hash(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=digest_size).hexdigest()


def build_sample_id(question: str, final_answer: str) -> str:
    key = f"{normalize_text_for_hash(question)}||{normalize_text_for_hash(final_answer)}"
    return stable_hash(key, digest_size=10)


def stable_bucket(sample_id: str, *, modulo: int = 10_000) -> int:
    return int(sample_id[:8], 16) % modulo
