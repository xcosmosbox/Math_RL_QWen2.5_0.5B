from __future__ import annotations


INSTRUCTION_TEMPLATE = """You are solving a math problem.

Problem:
{problem}

Requirements:
1. Show concise step-by-step reasoning.
2. End with exactly one final line in the format: Final Answer: <answer>
3. Do not omit the final answer line.
"""


def build_math_instruction_prompt(problem: str) -> str:
    return INSTRUCTION_TEMPLATE.format(problem=problem.strip())
