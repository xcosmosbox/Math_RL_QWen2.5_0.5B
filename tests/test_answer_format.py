from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.answer_extraction import extract_final_answer
from rewards.grpo_rewards import compute_reward
from training.prompt_templates import build_math_rl_instruction_prompt


def test_rl_prompt_template_uses_final_answer_line() -> None:
    prompt = build_math_rl_instruction_prompt("2+2=?")

    assert "Final Answer: <answer>" in prompt
    assert "<answer>your_final_answer</answer>" not in prompt
    assert "\\boxed{" not in prompt


def test_eval_task_prompts_use_final_answer_line() -> None:
    task_files = [
        PROJECT_ROOT / "eval/tasks/gsm8k_cot_mathrl/gsm8k_cot_mathrl.yaml",
        PROJECT_ROOT / "eval/tasks/hendrycks_math500_mathrl/hendrycks_math500_mathrl.yaml",
        PROJECT_ROOT / "eval/tasks/theoremqa_mathrl/theoremqa_mathrl.yaml",
    ]

    for task_file in task_files:
        content = task_file.read_text(encoding="utf-8")
        assert "Final Answer: <answer>" in content, task_file
        assert "<answer>your_final_answer</answer>" not in content, task_file
        assert "\\boxed{" not in content, task_file


def test_extraction_accepts_final_answer_line_without_tag() -> None:
    result = extract_final_answer("reasoning\nFinal Answer: 42")

    assert result.extracted_answer == "42"
    assert result.extraction_method == "final_answer_line"
    assert result.format_pass is True


def test_bare_answer_tag_is_parseable_but_not_format_pass() -> None:
    result = extract_final_answer("reasoning\n<answer>42</answer>")

    assert result.extracted_answer == "42"
    assert result.extraction_method == "answer_tag"
    assert result.format_pass is False


def test_reward_no_longer_adds_answer_tag_bonus() -> None:
    result = compute_reward(
        sample_id="case-1",
        completion="reasoning\nFinal Answer: 42",
        target_final_answer="42",
    )

    assert result.exact_match_pass is True
    assert result.format_pass is True
    assert result.format_reward > 0.0
    assert result.answer_tag_reward == 0.0
