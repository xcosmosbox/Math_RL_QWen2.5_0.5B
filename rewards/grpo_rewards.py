from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rewards.answer_extraction import extract_final_answer
from rewards.math_verifier import exact_match


CORRECTNESS_REWARD_VALUE = 1.0
FORMAT_REWARD_VALUE = 0.1


@dataclass(frozen=True)
class RewardResult:
    sample_id: str
    correctness_reward: float
    format_reward: float
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
    is_correct = exact_match(extraction.extracted_answer, target_final_answer)
    correctness_reward = CORRECTNESS_REWARD_VALUE if is_correct else 0.0
    format_reward = FORMAT_REWARD_VALUE if extraction.format_pass else 0.0
    total_reward = correctness_reward + format_reward
    return RewardResult(
        sample_id=sample_id,
        correctness_reward=correctness_reward,
        format_reward=format_reward,
        total_reward=total_reward,
        extracted_answer=extraction.extracted_answer,
        format_pass=extraction.format_pass,
        raw_completion=completion,
        drop_or_parse_flags=list(extraction.parse_flags),
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
