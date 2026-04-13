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
