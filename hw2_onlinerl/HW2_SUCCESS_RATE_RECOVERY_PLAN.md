# CS224R HW2 成功率达标计划

> 状态：Phase 1 已完成，正式 50-episode CSV 已同步到 `hw2_submission/`。  
> 目标：PPO 在 1M 附近达到 25%；3A 在 100k 前达到 90%；3B 在 40k 前达到 90%。  
> 边界：本机不运行 Meta-World、MuJoCo、训练或渲染；训练始终 `save_video=false`。不由生成式模型编写或修改三份作业算法文件。  
> 判据：只看正式 `eval/episode_success`，不使用 `viz/` 的 3-rollout 抽样。

---

## Phase 1：正式结果归档（已完成）

证据目录：`hw2_submission/`。

| 方法 | 正式结果 | 作业门槛 | 状态 |
|---|---:|---:|---|
| PPO | 18% @ 998k；全程最高 40% @ 890k | 1M 附近 ≥25% | 未达标 |
| 3A | 100k 前最高 64% @ 56k；100k 为 48% | 100k 前 ≥90% | 未达标 |
| 3B | 40k 前最高 26% @ 24k；40k 为 0% | 40k 前 ≥90% | 未达标 |

Archive 也未达标：PPO 最后 22%，3A 门槛内最高 62%，3B 门槛内最高 28%。

- [x] 三份正式 `eval.csv` 已同步
- [x] 三份 Hydra config/overrides 已同步
- [x] 主 run 与 archive 分开保存
- [x] 确认每个正式点包含 50 条 eval episode
- [x] 确认 `viz` 的 3/3 属于小样本噪声

### 从曲线得到的诊断

1. PPO 在 BC 后已经接近门槛：主 run 的 frame 0 为 26%，但训练后最后降到 18%。两轮 PPO 最后 10 个正式点的均值均约为 20.6%。问题更像训练阶段削弱了 BC 策略，而不是完全没有学会任务。
2. 3A 和 3B 最终都能学会：3A 首次 90% 在 250k，3B 首次 90% 在 144k。环境、奖励、demo 和 replay 主链路没有完全失效，但学习明显过晚。
3. 两次相同配置/seed 的 off-policy 曲线高度一致：3A 相关系数约 0.95，3B 约 0.92。单纯换 seed 不足以解释距离门槛的巨大差距。
4. 3B 确实比 3A 更快，但仍未达到作业要求的 sample efficiency；应先做受控的优化速度实验，再用新 seed 复验胜出的配置。

---

## Phase 2：重跑前补齐诊断证据

### 2.1 从 W&B 导出训练指标

已有 run ID：

- PPO：`xgv98hup`
- 3A：`14n6qslz`
- 3B：`t4ulxxlu`

从 W&B 分别导出以下 raw CSV；关闭 smoothing：

PPO：

- `pretrain/pretrain_actor_loss`
- `pretrain/actor_std`
- `actor/policy_loss`
- `actor/value_loss`
- `actor/entropy`
- `actor/reverse_kl`
- `actor/actor_std`
- `actor/raw_advantage_mean`
- `actor/raw_advantage_std`

3A / 3B：

- `pretrain/pretrain_actor_loss`
- `actor/actor_loss`
- `critic/critic_loss`
- `critic/bellman_target`
- `critic/q_online`

保存到：

```text
hw2_submission/diagnostics/
├── ppo/
├── 3a/
└── 3b/
```

- [ ] PPO 诊断指标已导出
- [ ] 3A 诊断指标已导出
- [ ] 3B 诊断指标已导出
- [ ] 确认没有 NaN/Inf
- [ ] 记录 PPO 的 actor std、reverse-KL 和 value loss 是否在后期异常漂移
- [ ] 比较 3A/3B 的 Q 值、Bellman target 和 critic loss 是否只是学习速度不同

若服务器 run 目录里存在 CSV，可先复制；不存在就以 W&B 导出为准：

```bash
conda activate cs224r-hw2-local
cd ~/ws/stanford_deep_rl/hw2_onlinerl/hw2\ 4

find Logdir/run_195512_device=cuda,save_snapshot=true,save_video=false,wandb_group=retrain_process -maxdepth 1 -type f -name '*.csv' -print
find Logdir/run_195513_device=cuda,save_snapshot=true,save_video=false,wandb_group=retrain_process -maxdepth 1 -type f -name '*.csv' -print
find Logdir/run_195513_agent.num_critics=10,device=cuda,save_snapshot=true,save_video=false,utd=5,wandb_group=retrain_process -maxdepth 1 -type f -name '*.csv' -print
```

### 2.2 核对训练代码版本

在服务器执行并保存输出：

```bash
conda activate cs224r-hw2-local
cd ~/ws/stanford_deep_rl
git rev-parse HEAD
git status --short
git diff -- hw2_onlinerl/hw2\ 4/on_policy.py hw2_onlinerl/hw2\ 4/off_policy.py hw2_onlinerl/hw2\ 4/gridworld_q_learning.py
```

- [ ] 服务器 commit 与本机预期一致
- [ ] 三份算法文件没有未记录差异
- [ ] 非 `YOUR CODE HERE` 的算法骨架与课程 starter 一致
- [ ] `python -m pytest tests/test_on_policy.py -k compute_gae` 通过

### 2.3 当前静态检查结论

已对本机代码和 student commit 做过只读检查：

- PPO 的 GAE 反向累计、done mask、无梯度 value target、log-prob ratio 和 clipped policy loss 均与当前 `HW2_TODO.md` 的要求对齐。
- Off-policy 的 BC、下一状态动作采样、两个 target critic 的较小值、全部 online critic 的 loss、软更新和 actor 的平均 Q 目标均已写入对应位置。
- P1 的 epsilon-greedy、greedy action 和 Q-learning 内环已写入；它与 Meta-World 的 P2/P3 成功率无关。
- 因此当前没有发现一个能单独解释 P2/P3 失败的明显 `YOUR CODE HERE` 公式错误。由于本机没有 Meta-World，尚未做真实环境的动态集成验证。

需要人工核对、但本轮不自动修改的可疑点：

- `on_policy.py` 的 `Actor.__init__` 接收 `log_std_init`，但没有使用它初始化 `self.log_std`；`PPOAgent` 传入的 `std=0.1` 因而不会按参数生效。这个问题已存在于 starter 骨架的非占位代码中，先与课程 starter/PDF 核对，不直接改算法文件。
- `viz_progress.py` 的 `rollout_eval()` 会调用全局 `np.random.seed()` 和 `torch.manual_seed()`，且把 agent 切到 eval mode 后不恢复。若训练时启用了 `HW2_PROGRESS_METHOD`，可视化会改变后续训练随机数状态。恢复训练时必须 unset 这两个可视化变量；这不是专家 mp4 标签问题。
- `off_policy.py` 的 actor update 会经过 critic 计算 actor 梯度，但只执行 actor optimizer；正常训练顺序下不会更新 Q 参数，不过会产生无用 critic 梯度。这是效率/实现卫生问题，不是目前首要的 sample-efficiency 解释。
- logger 当前会把 actor/critic/pretrain 指标送到 W&B，但训练循环没有对应的 CSV dump；因此诊断指标优先从 W&B 导出，不能因为本地缺少这些 CSV 就判断训练没记录。

- [x] 已完成 student commit 的静态 diff 检查
- [x] 已识别非算法可视化对随机性的污染风险
- [ ] 在服务器/学生自己的含 pytest 环境中完成 GAE 单测

### Phase 2 决策门

- 若出现 NaN/Inf、损失爆炸或代码版本不一致：先停止，定位原因，不开始 sweep。
- 若指标有限、代码和配置一致：进入 Phase 3。

---

## Phase 3：六卡受控补跑

这轮不改源文件。每项实验只改变一个主要因素，并保留独立 W&B 名称。关闭进度可视化变量，避免覆盖现有 `viz/`，也避免评估过程改变训练随机数状态。

### 3.1 服务器统一准备

```bash
conda activate cs224r-hw2-local
unset LD_LIBRARY_PATH CPATH LIBRARY_PATH
source ~/.local/hw2-sysdeps/env.sh
cd ~/ws/stanford_deep_rl/hw2_onlinerl/hw2\ 4
unset HW2_PROGRESS_METHOD HW2_PROGRESS_VIZ_ROOT
mkdir -p recovery_logs
```

### 3.2 GPU 0：PPO 默认配置，新 seed

目的：测量 seed 方差，不改 PPO 超参。

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=hw2-recovery-ppo-default-seed3 python train_on_policy.py device=cuda save_video=false save_snapshot=true seed=3 wandb_group=hw2_recovery_ppo 2>&1 | tee recovery_logs/ppo_default_seed3.log
```

验收：跑满 1M，最后一个正式点 ≥25%。

### 3.3 GPU 1：PPO 保留 BC 锚点

目的：主 run 在 frame 0 已达 26%，但后期降至 18%；此实验只启用训练期间的间歇 BC。`bc_freq=1` 在当前训练循环中表示每次 PPO rollout update 后做一次 BC update。

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_NAME=hw2-recovery-ppo-bc-seed2 python train_on_policy.py device=cuda save_video=false save_snapshot=true seed=2 bc_freq=1 wandb_group=hw2_recovery_ppo 2>&1 | tee recovery_logs/ppo_bc_seed2.log
```

验收：跑满 1M，最后一个正式点 ≥25%；同时检查后期没有只靠单点偶然越线。

### 3.4 GPU 2–3：3A 学习率单因素实验

依据：当前 3A 到 250k 才首次达到 90%，需要约 2.5 倍的样本效率提升。保持 seed=1、UTD=1、2 critics 和其他配置不变，只改变共享学习率。

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_NAME=hw2-recovery-3a-lr3e-4 python train_off_policy.py device=cuda save_video=false save_snapshot=true seed=1 agent.num_critics=2 utd=1 lr=3e-4 num_train_frames=120000 wandb_group=hw2_recovery_3a 2>&1 | tee recovery_logs/3a_lr3e-4.log
```

```bash
CUDA_VISIBLE_DEVICES=3 WANDB_NAME=hw2-recovery-3a-lr5e-4 python train_off_policy.py device=cuda save_video=false save_snapshot=true seed=1 agent.num_critics=2 utd=1 lr=5e-4 num_train_frames=120000 wandb_group=hw2_recovery_3a 2>&1 | tee recovery_logs/3a_lr5e-4.log
```

验收：100k 以内至少一个正式 50-episode raw point ≥90%。若两者都达标，优先保留更低的 `3e-4`。

### 3.5 GPU 4–5：3B 学习率单因素实验

依据：当前 3B 到 144k 才首次达到 90%，需要约 3.6 倍的样本效率提升。保持 seed=1、UTD=5、10 critics 和其他配置不变。

```bash
CUDA_VISIBLE_DEVICES=4 WANDB_NAME=hw2-recovery-3b-lr3e-4 python train_off_policy.py device=cuda save_video=false save_snapshot=true seed=1 agent.num_critics=10 utd=5 lr=3e-4 num_train_frames=60000 wandb_group=hw2_recovery_3b 2>&1 | tee recovery_logs/3b_lr3e-4.log
```

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_NAME=hw2-recovery-3b-lr5e-4 python train_off_policy.py device=cuda save_video=false save_snapshot=true seed=1 agent.num_critics=10 utd=5 lr=5e-4 num_train_frames=60000 wandb_group=hw2_recovery_3b 2>&1 | tee recovery_logs/3b_lr5e-4.log
```

验收：40k 以内至少一个正式 50-episode raw point ≥90%。若两者都达标，优先保留更低的 `3e-4`。

### 3.6 运行检查

- [ ] 六条命令均包含 `device=cuda save_video=false`
- [ ] 没有设置 `HW2_PROGRESS_METHOD`
- [ ] W&B run 名清楚标识方法、配置和 seed
- [ ] 每 2k frames 产生一个 50-episode 正式点
- [ ] PPO 跑满 1M
- [ ] 3A 至少跑过 100k
- [ ] 3B 至少跑过 40k
- [ ] 保存每条 run 的 `eval.csv`、Hydra config、overrides 和 W&B URL

---

## Phase 4：胜出配置复验

### 4.1 PPO

- 默认 seed 3 达标：再检查最后 5–10 个点是否稳定，不需要立刻增加正则。
- 间歇 BC 达标：用 seed 3 再跑一次 `bc_freq=1`，确认不是 seed 2 偶然结果。
- 两者都不达标：先结合 Phase 2 的 actor std、reverse-KL 和 value loss 判断，再考虑单独提高 reverse-KL 系数；不要同时改多个 PPO 超参。

### 4.2 3A / 3B

- 某档学习率达标：用 seed 3 复跑同一档，确认 sample efficiency 能复现。
- 两档都未达标且损失稳定：不要继续盲换 seed；先检查 actor/critic 指标和课程 starter 差异。
- 出现损失爆炸：放弃该档，保留较低学习率结果，不把偶然峰值作为最终 run。

复验时继续保持作业指定的 critics/UTD：

- 3A 必须是 2 critics、UTD=1。
- 3B 必须是 10 critics、UTD=5。

---

## Phase 5：正式归档与报告

每种方法选择一条配置正确且正式达标的 run，替换 `hw2_submission/CSV files/` 中对应的旧 CSV，同时把旧文件留在 `runs/archive/`。

- [ ] PPO 最后一个 1M 附近 raw point ≥25%
- [ ] 3A 在 100k 内出现 ≥90% raw point
- [ ] 3B 在 40k 内出现 ≥90% raw point
- [ ] 截图 smoothing=0
- [ ] PPO 横轴显示到 1M
- [ ] 3A 横轴显示到 100k
- [ ] 3B 横轴显示到 40k
- [ ] 报告记录实际 seed 和所有非默认 Hydra overrides
- [ ] CSV、config、overrides、W&B URL 一一对应
- [ ] 不覆盖或删除失败 run

### 最终记录区

| 方法 | 配置 | seed | W&B URL | 作业窗口内结果 | 达标 |
|---|---|---:|---|---:|---|
| PPO |  |  |  |  |  |
| 3A |  |  |  |  |  |
| 3B |  |  |  |  |  |

---

## 最短执行顺序

```text
导出 W&B 训练诊断指标
  → 排除 NaN、代码版本和配置问题
  → 六卡并行跑 2 个 PPO + 2 个 3A + 2 个 3B 受控实验
  → 用正式 50-episode CSV 判定
  → 胜出配置换 seed 复验
  → 更新 hw2_submission 与报告
```

*更新日期：2026-09-10*
