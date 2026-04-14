from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from rewards.answer_extraction import ANSWER_TAG_RE, extract_final_answer, trim_completion_to_answer
from rewards.math_verifier import exact_match, equivalent_match_offline


EXACT_MATCH_REWARD_VALUE = 1.0
EQUIVALENT_MATCH_REWARD_VALUE = 0.8
FORMAT_REWARD_VALUE = 0.05
ANSWER_TAG_REWARD_VALUE = 0.08
CLEAN_STOP_REWARD_VALUE = 0.03
NO_ANSWER_PENALTY_VALUE = -0.12
REPEAT_PENALTY_VALUE = -0.08
ROLE_MARKER_PENALTY_VALUE = -0.08
TAIL_PENALTY_PER_TOKEN = -0.015
MAX_TAIL_PENALTY = -0.20
OVERLENGTH_TOKEN_BUDGET = 256
OVERLENGTH_PENALTY_PER_TOKEN = -0.0005
MAX_OVERLENGTH_PENALTY = -0.05
REPEATED_LINE_THRESHOLD = 2
TOKEN_SPLIT_RE = re.compile(r"\S+")
ROLE_MARKER_RE = re.compile(r"(?i)(?:^|\n)\s*(?:human|assistant|user)\s*:")


@dataclass(frozen=True)
class RewardResult:
    sample_id: str
    correctness_reward: float
    exact_match_pass: bool
    equivalent_match_pass: bool
    format_reward: float
    answer_tag_reward: float
    clean_stop_reward: float
    no_answer_penalty: float
    repeat_penalty: float
    role_marker_penalty: float
    tail_penalty: float
    overlength_penalty: float
    total_reward: float
    extracted_answer: str | None
    format_pass: bool
    raw_completion: str
    drop_or_parse_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_reward(
    *,
    sample_id: str,
    completion: str,
    target_final_answer: str,
) -> RewardResult:
    extraction = extract_final_answer(completion)
    exact_match_pass = exact_match(extraction.extracted_answer, target_final_answer)
    equivalent_match_pass = equivalent_match_offline(extraction.extracted_answer, target_final_answer)
    if exact_match_pass:
        correctness_reward = EXACT_MATCH_REWARD_VALUE
    elif equivalent_match_pass:
        correctness_reward = EQUIVALENT_MATCH_REWARD_VALUE
    else:
        correctness_reward = 0.0

    format_reward = FORMAT_REWARD_VALUE if extraction.format_pass else 0.0
    answer_tag_reward = ANSWER_TAG_REWARD_VALUE if ANSWER_TAG_RE.search(completion or "") else 0.0

    trimmed_completion, trim_flags = trim_completion_to_answer(completion)
    extra_tail = (completion or "")[len(trimmed_completion) :].strip()
    has_tail = bool(extra_tail)
    clean_stop_reward = CLEAN_STOP_REWARD_VALUE if extraction.extracted_answer is not None and not has_tail else 0.0

    no_answer_penalty = NO_ANSWER_PENALTY_VALUE if extraction.extracted_answer is None else 0.0
    tail_token_count = len(TOKEN_SPLIT_RE.findall(extra_tail))
    tail_penalty = max(MAX_TAIL_PENALTY, TAIL_PENALTY_PER_TOKEN * tail_token_count) if has_tail else 0.0
    role_marker_penalty = ROLE_MARKER_PENALTY_VALUE if ROLE_MARKER_RE.search(extra_tail) else 0.0

    normalized_lines = [line.strip() for line in (completion or "").splitlines() if line.strip()]
    repeated_line_count = max((normalized_lines.count(line) for line in set(normalized_lines)), default=0)
    repeat_penalty = REPEAT_PENALTY_VALUE if repeated_line_count >= REPEATED_LINE_THRESHOLD else 0.0

    completion_token_count = len(TOKEN_SPLIT_RE.findall(completion or ""))
    overflow = max(0, completion_token_count - OVERLENGTH_TOKEN_BUDGET)
    overlength_penalty = max(MAX_OVERLENGTH_PENALTY, OVERLENGTH_PENALTY_PER_TOKEN * overflow) if overflow else 0.0

    total_reward = (
        correctness_reward
        + format_reward
        + answer_tag_reward
        + clean_stop_reward
        + no_answer_penalty
        + repeat_penalty
        + role_marker_penalty
        + tail_penalty
        + overlength_penalty
    )
    return RewardResult(
        sample_id=sample_id,
        correctness_reward=correctness_reward,
        exact_match_pass=exact_match_pass,
        equivalent_match_pass=equivalent_match_pass,
        format_reward=format_reward,
        answer_tag_reward=answer_tag_reward,
        clean_stop_reward=clean_stop_reward,
        no_answer_penalty=no_answer_penalty,
        repeat_penalty=repeat_penalty,
        role_marker_penalty=role_marker_penalty,
        tail_penalty=tail_penalty,
        overlength_penalty=overlength_penalty,
        total_reward=total_reward,
        extracted_answer=extraction.extracted_answer,
        format_pass=extraction.format_pass,
        raw_completion=completion,
        drop_or_parse_flags=sorted(set(list(extraction.parse_flags) + trim_flags)),
    )


def batch_compute_rewards(
    samples: list[dict[str, str]],
    completions: list[str],
) -> list[RewardResult]:
    if len(samples) != len(completions):
        raise ValueError("samples and completions must have the same length")
    return [
        compute_reward(
            sample_id=sample["sample_id"],
            completion=completion,
            target_final_answer=sample["target_final_answer"],
        )
        for sample, completion in zip(samples, completions)
    ]
