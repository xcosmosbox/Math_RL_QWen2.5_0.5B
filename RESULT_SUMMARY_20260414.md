# 当前结果汇总

## 本次关注的主线

- 目标是把 `verl + GRPO + sglang` 这条线在远端单卡环境里跑起来。
- 当前使用的实验脚本是：
  `verl_bridge/run_verl_grpo_qwen25_15b_sglang_full_sft_b512_dyn24k.sh`
- 当前远端提交脚本是：
  `verl_bridge/submit_ais_qwen25_15b_grpo_full_sft_b512_dyn24k.sh`

## 当前已经确认的结果

### 1. 远端任务已经越过前面几次失败点

这次远端任务不是一开始就退出，而是已经进入训练主流程。

当前运行中的远端任务：

- `run_id`
  `1674707`
- `AIS run name`
  `run-k2tt6l`
- `status`
  `Training`

训练日志里已经确认出现：

- `Total training steps: 159`
- `Training Progress: 0/159`
- `swanlab: Tracking run with swanlab version 0.7.15`
- `swanlab: Syncing run ... to the cloud`

这说明下面几件事已经通过：

- `Ray` 集群可以正常拉起并连接
- `TaskRunner` 可以继续往下执行
- 数据集读取和长度过滤已经完成
- `SGLangHttpServer` 已启动
- `SwanLab` 已成功初始化并开始同步

### 2. 当前远端实验使用的核心配置

- 底座模型使用的是已经合并好的 `SFT merged` 模型
- `LoRA` 已关闭
- `rollout` 后端使用 `sglang`
- `attention backend` 使用 `triton`
- `train_batch_size=512`
- `ppo_mini_batch_size=128`
- 动态批大小已经开启
- `actor/ref/rollout` 的动态 token 上限为 `24000`
- `ref param offload=false`
- 当前远端这条 `GRPO` 线里 `actor param offload=true`

## 这轮排查里已经修住的问题

### 1. 远端训练脚本误停 `Ray`

前面的失败原因之一是：

- 远端提交器先启动了 `Ray head`
- 训练脚本里又执行了一次
  `ray stop --force`
- 后面 `main_ppo` 再去连接 `RAY_ADDRESS` 时，就会出现 `GCS` 连接超时

这部分已经处理为：

- 在远端 `GRPO` 包装脚本里补了
  `SKIP_RAY_STOP=1`

对应脚本：

- `verl_bridge/run_verl_grpo_qwen25_15b_sglang_full_sft_b512_dyn24k.sh`

### 2. `Ray runtime_env` 环境变量类型错误

另一处失败原因是：

- `VERL_DISABLE_TORCHAO=1` 被当成整数传给了 `Ray runtime_env`
- `Ray` 要求 `env_vars` 里的值必须是字符串

报错原文是：

- `runtime_env['env_vars'] must be of type Dict[str, str]`

这部分已经处理为：

- 把 `VERL_DISABLE_TORCHAO` 改成显式字符串传入

对应脚本：

- `verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh`

### 3. `torchao` 导致的远端导入失败

更早一轮远端报错来自：

- `transformers`
  误探测 `torchao`
- 进入 `torch._dynamo.device_interface`
  后出现设备属性索引异常

这部分已经处理为：

- 在仓库根目录增加了
  `sitecustomize.py`
- 通过
  `VERL_DISABLE_TORCHAO=1`
  屏蔽 `torchao` 探测

### 4. `SwanLab` 无密钥导致初始化失败

上一轮真正把训练打断的原因是：

- 远端容器里没有拿到
  `SWANLAB_API_KEY`
- `swanlab.init` 时抛出：
  `swanlab.error.KeyFileError: api key not configured (no-tty)`

这部分已经处理为：

- 提交脚本会优先检查当前环境里的
  `SWANLAB_API_KEY`
- 如果没有，再从本地
  `~/.swanlab/.netrc`
  读取并导出
- 再通过 `AIS` 的环境变量透传到远端

对应脚本：

- `verl_bridge/submit_ais_qwen25_15b_grpo_full_sft_b512_dyn24k.sh`

## 当前仍然观察到的现象

### 1. `sglang` 会打印一条 `triton` 回退提示

日志里仍然能看到：

- `Triton is not supported on current platform, roll back to CPU`

这条提示目前还没有直接导致任务退出，因为训练已经继续往后走，并且 `SGLang`、`SwanLab`、训练进度条都已经出现。

当前更像是：

- 某些局部算子没有使用 `Triton`
- 主训练流程本身还能继续运行

### 2. 远端节点在起新任务前会清理旧的 `Ray` 残留

日志里可以看到上一轮残留 `Ray` 进程被清理，包括一些 `zombie` 状态的进程。

目前这一步虽然比较吵，但这次没有挡住后续训练启动。

## 相关日志位置

### 当前运行中的远端训练日志

- `logs/20260414_201901_qwen25_15b_grpo_full_sftmerged_sglang_b512_dyn24k_u08.log`

### 当前运行中的远端提交器日志

- `logs/ais_qwen25_15b_grpo_dyn24k/qwen25_15b_grpo_full_sftmerged_sglang_b512_dyn24k_u08_rank0_20260414_201713.log`

### 上一轮因为 `SwanLab` 密钥缺失失败的日志

- `logs/20260414_200425_qwen25_15b_grpo_full_sftmerged_sglang_b512_dyn24k_u08.log`

## 当前涉及的主要脚本

- `verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh`
- `verl_bridge/run_verl_grpo_qwen25_15b_sglang_full_sft_b512_dyn24k.sh`
- `verl_bridge/submit_ais_qwen25_15b_grpo_full_sft_b512_dyn24k.sh`
- `sitecustomize.py`

## 当前可以怎样理解

这轮远端 `GRPO` 主线已经从“起不来”推进到了“已经开始训练”。

前面已经确认修住的几类问题包括：

- `Ray` 被误停
- `runtime_env` 类型错误
- `torchao` 导入链路异常
- `SwanLab` 远端无密钥

当前这条线还在继续运行，说明本轮排查至少已经把入口阶段和日志器初始化阶段稳定下来。
