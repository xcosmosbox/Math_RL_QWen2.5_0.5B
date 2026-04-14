# 当前分支说明

## 分支

`feat/bigmath-verified-pipeline`

## 这条分支目前做了什么

### 1. 数据清洗和导出

- 补了 `Big-Math-RL-Verified` 相关清洗和切分流程。
- 统一了 `SFT` 与 `GRPO` 导出逻辑，导出时都会带项目内统一的数学提示词。
- `SFT` 导出会清理旧的 `Final Answer` 块、去掉多余 `\boxed{}` 包装、尽量保留推理正文。
- `GRPO` 导出已经切到答案标签提示词格式：
  `<answer>\boxed{your_final_answer}</answer>`
- 新增了专门给 `RL` 用的 `RL-ready` 数据构建链路：
  `data/build_rl_ready_dataset.py`
- 这条链路不再要求 `chosen_solution` 存在，只要求题目和最终答案有效，因此把原来
  `no_usable_chosen_solution`
  卡掉的大量样本重新纳回了 `RL` 数据池。

### 2. SFT 训练链路

- `training/run_sft.py` 支持命令行覆盖：
  - `base_model`
  - `template`
  - `batch size`
  - `gradient accumulation`
  - `epoch`
  - `output_dir`
- 主线实测里，当前最稳的底座是：
  `Qwen2.5-1.5B-Instruct`
- 推荐的 `SFT` 产物：
  `outputs/sft/strict_main_qwen25_15b_bs32_ga1`

### 3. 答案抽取和判分

- `rewards/answer_extraction.py` 现在同时兼容：
  - `Final Answer: ...`
  - `<answer>...</answer>`
  - `\boxed{...}`
- 新增了答案锚点截断逻辑，能把答案后的拖尾切掉。
- 支持一些常见等价写法：
  - `d / D`
  - `20√13 / 20\sqrt{13}`
  - 数字里的逗号去掉后再比较
- `rewards/math_verifier.py` 放宽了简单等价匹配。

### 4. TRL / GRPO 奖励函数

- `rewards/grpo_rewards.py` 已扩成多项奖励：
  - 正确性奖励
  - 等价正确奖励
  - 格式奖励
  - `<answer>` 标签奖励
  - 干净收尾奖励
  - 无答案惩罚
  - 重复惩罚
  - 角色串惩罚
  - 拖尾惩罚
  - 超长惩罚
- `analysis/build_grpo_reward_summary.py` 已补这些新奖励项的汇总。

### 5. TRL / GRPO 训练脚本

- `training/run_grpo.py` 支持命令行覆盖：
  - `base_sft_model_path`
  - `output_dir`
  - `train batch`
  - `generation batch`
  - `num_generations`
  - `max completion length`
  - `temperature`
  - `top_p`
  - `repetition_penalty`
  - 其他训练超参

## 已验证过的结果

### SFT

- `Qwen2.5-1.5B-Instruct` 全量 `SFT` 已完成。
- 当前更推荐继续拿这版做 `RL` 底座。

### TRL smoke RL

- 已连续跑过多轮 `smoke`，关键奖励信号已经比最初更干净。
- 当前最好的 `smoke` 结果：
  `outputs/grpo/strict_smoke_qwen25_15b_b16_gb16_g4_v3`
- `TRL` 这条线当前仍然是最稳的 `RL` 主线备选。

### verl 迁移

- 已补桥接层：
  - `verl_bridge/export_verl_grpo_dataset.py`
  - `verl_bridge/reward_fn.py`
  - `verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh`
- `jsonl -> parquet` 转换已跑通。
- `verl` 的自定义奖励函数已能直接复用当前项目里的奖励逻辑。
- `Search-R1-Qwen3/.venv` 里已确认有：
  - `verl`
  - `ray`
  - `hydra`
  - `vllm`
  - `swanlab`
- `Search-R1-Qwen3/.venv-sglang-migrated` 里已确认有：
  - `verl`
  - `ray`
  - `hydra`
  - `sglang`
  - `vllm`
  - `swanlab`
- `sglang smoke` 已经跑到比 `vllm` 更深的阶段：
  - `FSDP actor` 初始化成功
  - `SGLangHttpServer` 成功启动
  - `SwanLab` run 成功创建并开始同步

## 当前外部环境修改

以下改动不在本仓库内，但已经做过：

- 在 `Search-R1-Qwen3/.venv` 里安装了 `swanlab`
- 在外部仓库
  `/home/work/opa_common_mx/wsy/Search-R1-Qwen3/verl`
  里改了：
  `verl/trainer/constants_ppo.py`

这个改动的作用是把下面这些变量转发进 `Ray worker`：

- `SWANLAB_API_KEY`
- `SWANLAB_MODE`
- `SWANLAB_LOG_DIR`

## 当前还没有完全收住的问题

### 1. TRL 主线和 verl 主线并存

- 目前项目内同时保留了：
  - `TRL + GRPO`
  - `verl + GRPO`
- 还没有完全收成单一路线。

### 2. verl + vLLM 仍有启动稳定性问题

- 用 `LoRA + vLLM` 时，之前出现过权重同步阶段的非法显存访问。
- 改成非 `LoRA` 后，`verl` 主链路能走得更远，但 `vLLM` 仍然对显存参数比较敏感。
- 当前结论是：
  单卡 `verl` 能跑，但参数要保守，不能直接照多卡示例上。
- 更具体地说，单独 `vLLM` 推理是正常的，问题集中在：
  `verl + vllm + 权重热更新`
  这条链路。

### 3. SwanLab 线上记录

- `swanlab login -k` 已成功。
- 之前线上没有记录，根因不是登录失败，而是：
  `Ray worker` 没有拿到 `SWANLAB_API_KEY`
- 这个问题已经在外部 `verl` 仓库里补过环境变量转发。
- 当前 `sglang smoke` 已经确认会打印 `SwanLab` 项目链接和 run 链接。

### 4. RL 主线数据口径已经改变

- 旧的严格 `GRPO` 主线只有：
  `1298 / 169`
- 新的 `RL-ready strict` 已经扩到：
  `81463 / 10207`
- `RL-ready relaxed` 已经扩到：
  `81478 / 10208`
- 当前 `training/export_grpo_dataset.py` 已切到读取：
  - `data/processed/rl_strict_train.jsonl`
  - `data/processed/rl_strict_valid.jsonl`
  - `data/processed/rl_relaxed_train.jsonl`
  - `data/processed/rl_relaxed_valid.jsonl`

## 关键产物路径

### 推荐 SFT 底座

- `outputs/sft/strict_main_qwen25_15b_bs32_ga1`

### TRL smoke RL

- `outputs/grpo/strict_smoke_qwen25_15b_b16_gb16_g4`
- `outputs/grpo/strict_smoke_qwen25_15b_b16_gb16_g4_v2`
- `outputs/grpo/strict_smoke_qwen25_15b_b16_gb16_g4_v3`

### RL-ready 数据

- `data/processed/rl_strict.jsonl`
- `data/processed/rl_strict_train.jsonl`
- `data/processed/rl_strict_valid.jsonl`
- `data/processed/rl_relaxed.jsonl`
- `data/processed/rl_relaxed_train.jsonl`
- `data/processed/rl_relaxed_valid.jsonl`

### verl 桥接

- `verl_bridge/export_verl_grpo_dataset.py`
- `verl_bridge/reward_fn.py`
- `verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh`

### verl 数据

- `data/processed/verl/strict_main/train.parquet`
- `data/processed/verl/strict_main/valid.parquet`
- `data/processed/verl/strict_main/smoke/train.parquet`
- `data/processed/verl/strict_main/smoke/valid.parquet`

### verl / SwanLab

- `outputs/verl/qwen25_15b_grpo_smoke_sglang_single_gpu`
- `outputs/verl/qwen25_15b_lora_grpo_single_gpu`
- `outputs/verl/qwen25_15b_full_grpo_verl_single_gpu`

## 现在可以怎么理解

这条分支已经把三件关键事做出来了：

1. `SFT` 底座已经换到更稳的 `Qwen2.5-1.5B`
2. 奖励函数已经从单一正确性分，扩成了面向数学推理的多项奖励
3. `RL` 数据已经从“小而稳的严格子集”扩到了“只要求题目和最终答案有效”的大数据池
4. `verl` 迁移已经不再停留在想法层面，桥接数据、奖励函数和单卡启动脚本都已经落下来了

剩下主要是在把 `verl` 的单卡 rollout 后端彻底收稳。当前判断是：
`vllm` 这条热更新链路不稳，`sglang` 更值得继续推进。

## 2026-04-14 评测与结果记录

这一轮主要做了两件事：

1. 把评测链路整理成可以稳定复用的状态。
2. 先把 `Original`、`SFT`、`RL step20`、`RL final` 这几组模型的结果跑全，再分析格式、截断和提示词问题。

### 这一轮评测具体做了什么

#### 1. 评测任务补齐

这次把下面四个评测集统一纳入了当前项目内的评测脚本：

- `MATH-500`
- `GSM8K`
- `HellaSwag`
- `TheoremQA`

其中：

- `MATH-500` 和 `GSM8K` 之前已经有评测基础，这次主要补的是统一提示词和答案抽取。
- `HellaSwag` 主要用来观察数学强化训练之后，通用选择题能力有没有明显回落。
- `TheoremQA` 是后来补进来的，本仓库里原先没有现成任务文件，这次新加了本地任务定义和解析逻辑。

#### 2. 评测后端处理

原本尝试过把评测切到 `vllm`，但当前环境里的 `vllm` 和 `torch` 组合存在二进制兼容问题，实际跑起来报过动态库符号错误。

因此这一轮正式结果统一改用 `hf` 后端来跑。这样速度会慢一些，但结果稳定，可复现，也便于继续调提示词和解析逻辑。

#### 3. 数学类提示词统一

一开始，`MATH-500` 和 `GSM8K` 的评测提示词还是比较普通的问答格式，和训练时使用的数学提示词并不一致。后来把评测提示词统一成了下面这套结构：

- 开头固定说明这是在解一道数学题
- 要求给出简短推理
- 最后一行必须按
  `<answer>\boxed{your_final_answer}</answer>`
  的格式收尾

这一步做完之后，`Original`、`SFT`、`RL` 三组在 `MATH-500` 和 `GSM8K` 上的结果都有明显变化，说明前面的评测确实受到了提示词不一致的影响。

#### 4. 答案抽取放宽

仅靠严格匹配 `<answer>` 或严格匹配 `\boxed{}`，会漏掉不少本来已经答对的样本。

因此这次把数学评测的答案抽取放宽到下面几类都能识别：

- `<answer>...</answer>`
- `\boxed{...}`
- `Final Answer: ...`
- `The answer is ...`
- 最后一行的公式
- 简单等式右侧
- 一些 `\text{...}` 包裹形式

做完这一步后，`MATH-500` 的得分提升比较明显，尤其是 `Original` 和 `SFT` 两组。

#### 5. TheoremQA 解析修正

`TheoremQA` 第一次补跑时，评测并不是模型崩掉，而是解析逻辑自己报错了。

具体问题是：

- 某些输出里会出现非常大的数字，甚至会走到 `inf`
- 旧解析逻辑会把它强行当成整数去转
- 最后在评测阶段报 `OverflowError`

这次已经把 `TheoremQA` 的解析逻辑改成：

- 先把数值转成有限浮点数
- `nan`、`inf` 这一类值直接视为无效候选
- 列表题型也做同样处理

修正后，`Original TheoremQA` 已经可以正常跑完并生成结果文件。

### 当前已拿到的正式结果

这里先分成两类来看。

第一类是 `Base 原模板`。这表示旧的数学评测提示词，也就是还没有统一成当前训练提示词时的历史结果。

第二类是当前主线结果，也就是统一提示词之后的结果。

#### Base 原模板

| 模型 | MATH-500 | GSM8K | HellaSwag | TheoremQA |
| --- | ---: | ---: | ---: | ---: |
| Base 原模板 | 0.2480 | 0.5906 | 0.6826 | 暂无 |

这里的 `TheoremQA` 之所以还是空着，不是因为官方没有默认模板，而是当时项目里还没有把 `TheoremQA` 接进评测，所以没有留下这一口径下的历史结果。

#### 当前统一提示词结果

| 模型 | MATH-500 | GSM8K | HellaSwag | TheoremQA |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.4100 | 0.6596 | 0.6826 | 0.2490 |
| SFT | 0.4220 | 0.6558 | 0.6823 | 0.2396 |
| RL step20 | 0.4460 | 0.6725 | 0.6834 | 0.2517 |
| RL final | 0.4760 | 0.6884 | 0.6831 | 0.2396 |

从当前这组结果可以先得到几个判断：

1. `RL final` 在 `MATH-500` 和 `GSM8K` 上是当前最好的。
2. `HellaSwag` 整体非常平，说明这一轮训练没有让通用能力出现大幅回落。
3. `TheoremQA` 上反而是 `RL step20` 略高，`RL final` 没有继续上涨，说明后续训练未必继续强化了这类题。

### TheoremQA 当前发现的问题

虽然已经拿到分数，但这一项还没有完全看完。

这次专门检查了 `TheoremQA` 的输入长度和输出长度问题。

#### 1. 输入不会截断

当前统一提示词下，`TheoremQA` 的输入长度很短：

- 中位数大约 `118`
- `p99` 大约 `273`
- 最大值 `327`

而当前底座模型的上下文窗口是 `32768`，所以输入侧没有压力。

#### 2. 输出存在明确截断

之前 `TheoremQA` 的生成上限设成了 `1024`。把样本输出重新做了一遍分词统计之后，发现确实有不少样本正好撞到了这个上限。

当前统计如下：

| 模型 | 样本数 | 恰好 1024 token 的样本数 |
| --- | ---: | ---: |
| Base | 747 | 90 |
| SFT | 747 | 21 |
| RL step20 | 747 | 37 |
| RL final | 747 | 3 |

而且这些撞线样本里，绝大多数尾部都没有 `<answer>` 或 `\boxed{}`，这说明很多样本不是自然结束，而是被上限截断了。

这件事带来的直接影响是：

- `Base` 的 `TheoremQA` 成绩很可能被低估
- `SFT` 和 `RL step20` 也会有一定影响
- `RL final` 受影响相对较小

### 针对 TheoremQA 的后续处理

基于上面的截断统计，这次已经把 `TheoremQA` 的 `max_gen_toks` 从 `1024` 调到了 `2048`。

本次修改点有两处：

- `eval/config_presets.py`
- `eval/tasks/theoremqa_mathrl/theoremqa_mathrl.yaml`

这样处理之后：

- `MATH-500`、`GSM8K`、`HellaSwag` 仍然保持原来的生成上限
- 只有 `TheoremQA` 被单独放大
- 前后结果更容易比较

### 2048 上限重跑状态

`TheoremQA` 的新一轮重跑已经启动，当前先从 `Original` 开始顺序执行。

现阶段已经确认：

- 新评测命令里已经带上了 `max_gen_toks=2048`
- 运行日志里也已经显示本次确实按 `2048` 生效
- 第一条样本耗时明显变长，说明长输出已经继续生成，而不是像之前那样早早撞到 `1024`

这一轮重跑完成之后，需要重点重新看两件事：

1. `TheoremQA` 分数本身有没有变化
2. 恰好撞线的样本数是否明显下降

### 当前可以怎样理解这批结果

截至目前，这条分支已经把训练和评测的主线关系看得比较清楚了。

先说训练收益：

- `SFT` 对数学基准有帮助，但增幅有限
- `RL step20` 在数学能力上继续往上走
- `RL final` 在 `MATH-500` 和 `GSM8K` 上是当前最佳

再说评测口径：

- 数学题如果不把提示词统一到训练格式，结果会明显失真
- 答案抽取过严也会压低成绩
- `TheoremQA` 还存在输出截断问题，因此这一项当前分数可以先看作阶段性结果

也就是说，当前最稳的一版结论是：

1. `RL final` 已经明显优于 `Original` 和 `SFT`
2. `RL step20` 在 `TheoremQA` 上暂时更好
3. `TheoremQA` 还要等 `2048` 上限重跑完，才能做最终判断

### 这一轮新增和修改过的评测相关文件

#### 评测脚本

- `eval/run_lm_eval.py`
- `eval/config_presets.py`
- `eval/summarize_results.py`

#### 本地任务

- `eval/tasks/hendrycks_math500_mathrl/hendrycks_math500_mathrl.yaml`
- `eval/tasks/hendrycks_math500_mathrl/utils.py`
- `eval/tasks/gsm8k_cot_mathrl/gsm8k_cot_mathrl.yaml`
- `eval/tasks/gsm8k_cot_mathrl/utils.py`
- `eval/tasks/theoremqa_mathrl/theoremqa_mathrl.yaml`
- `eval/tasks/theoremqa_mathrl/utils.py`

#### 结果目录

- `eval/outputs/2026-04-14_base_math_hf`
- `eval/outputs/2026-04-14_sft_math_hf`
- `eval/outputs/2026-04-14_rl_math_hf`
- `eval/outputs/2026-04-14_step20_math_hf`
- `eval/outputs/2026-04-13_base_hellaswag_hf`
- `eval/outputs/2026-04-13_sft_hellaswag_hf`
- `eval/outputs/2026-04-13_rl_hellaswag_hf`
- `eval/outputs/2026-04-14_step20_hellaswag_hf`
- `eval/outputs/2026-04-14_base_theoremqa_hf`
- `eval/outputs/2026-04-14_sft_theoremqa_hf`
- `eval/outputs/2026-04-14_rl_theoremqa_hf`
- `eval/outputs/2026-04-14_step20_theoremqa_hf`
