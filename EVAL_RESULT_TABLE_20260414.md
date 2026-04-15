# 当前评测结果汇总

## 结果范围

这份文档整理的是当前 `1.5B` 相关模型已经完成的正式评测结果。

本次汇总纳入的阶段包括：

1. `qwen base`
2. `qwen base sft`
3. `grpo sampled24k`
4. `dapo`
5. `grpo`

旧版 `0.5B smoke` 结果仍然保留在仓库中，但这份表不再混放进去。

## 参与对比的模型

1. `qwen base`
   `Qwen/Qwen2.5-1.5B`
2. `qwen base sft`
   `outputs/sft/strict_main_qwen25_15b_base_bs32_ga1`
3. `grpo sampled24k`
   `qwen base sft + GRPO sampled24k`
4. `dapo`
   `qwen base sft + DAPO`
5. `grpo`
   `qwen base sft + GRPO`

## 主结果表

| 模型阶段 | 训练数据量 | GSM8K | MATH-500 | TheoremQA | HellaSwag acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen base | 0 | 0.0121 | 0.0140 | 0.0576 | 0.5018 |
| qwen base sft | 1298 | 0.1638 | 0.1440 | 0.1673 | 0.5068 |
| grpo sampled24k | 24000 | 0.6884 | 0.4760 | 0.2396 | 0.5101 |
| dapo | 81463 | 0.7028 | 0.4180 | 0.2691 | 0.5105 |
| grpo | 81463 | 0.7043 | 0.4380 | 0.2784 | 0.5112 |

## 当前结果说明

1. 原始 `Qwen/Qwen2.5-1.5B` 在这套数学指令格式上的分数很低，说明它本身没有学会当前要求的回答格式。
2. `qwen base sft` 已经把原始 `Base` 从 `GSM8K 0.0121 -> 0.1638`、`MATH-500 0.0140 -> 0.1440`、`TheoremQA 0.0576 -> 0.1673` 拉起来，`HellaSwag acc` 也从 `0.5018 -> 0.5068`。
3. `GSM8K` 当前最好的是 `grpo`，分数是 `0.7043`。
4. `MATH-500` 当前最好的是 `grpo sampled24k`，分数是 `0.4760`。
5. `TheoremQA` 当前最好的是 `grpo`，分数是 `0.2784`。
6. `HellaSwag acc` 当前最好的是 `grpo`，分数是 `0.5112`。

## 补充备注

1. 这份表统一使用当前仓库内已经完成并保存下来的 `hf` 评测结果。
2. `GSM8K` 使用的是当前 `mathrl` 任务配置，对应一套统一的答案抽取和归一化逻辑。
3. `dapo` 与 `grpo` 这两行对应的是中途检查点产物，并非完整训练结束点。
4. `qwen base` 这一行使用的是纯文本提示格式，没有套用 `chat template`。
5. `qwen base sft` 使用的是 `Qwen/Qwen2.5-1.5B + LoRA SFT adapter`，评测时沿用当前数学任务的 `chat template`。
6. `训练数据量` 这一列指的是当前阶段实际使用的训练集样本数，不包含验证集。
