# 当前评测结果汇总

## 结果范围

这份文档只整理当前仓库里已经汇总完成的 `eval` 结果。

- 数据来源：
  `eval/summaries/leaderboard.csv`
  `analysis/results/main_result_table.csv`
  `eval/summaries/regression_table.csv`
- 汇总日期：
  `2026-04-10`
- `lm-eval-harness`（评测框架）版本：
  `0.4.11`

## 参与对比的模型

- `base`：
  `Qwen/Qwen2.5-0.5B`
- `sft`：
  `outputs/sft/strict_smoke`
- `rl`：
  `outputs/grpo/strict_smoke/checkpoints/final`

## 主结果表

| 模型阶段 | GSM8K strict | GSM8K flexible | MATH-500 | HellaSwag acc | HellaSwag acc_norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.26 | 0.28 | 0.00 | 0.38 | 0.52 |
| sft | 0.36 | 0.26 | 0.00 | 0.40 | 0.52 |
| rl | 0.38 | 0.20 | 0.00 | 0.40 | 0.54 |

## 当前结果说明

1. `GSM8K strict-match` 上，`sft` 相比 `base` 提升 `0.10`，`rl` 相比 `sft` 再提升 `0.02`，当前最好结果是 `0.38`。
2. `HellaSwag` 基本维持住了原有水平，`rl` 在 `acc_norm` 上比 `base` 高 `0.02`。
3. `MATH-500` 当前三组结果都是 `0.00`，现有这批 `smoke` 评测还没有体现出提升。

## 补充备注

1. 当前汇总文件里有 `GSM8K`、`MATH-500`、`HellaSwag` 三项结果。
2. `eval/tasks` 目录里虽然已经有 `TheoremQA` 配置，但这份现成汇总里还没有对应分数。
3. 这份表对应的是仓库内已经落盘的历史评测结果，今天仍在运行的远端训练任务还没有进入这份表。
