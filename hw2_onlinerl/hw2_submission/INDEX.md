# HW2 正式成功率文件（50-episode eval.csv）

本目录从服务器 `Logdir/` 抽出**可同步**的评测曲线和 Hydra 配置。
`Logdir/` 仍被 gitignore（含 snapshot / wandb / replay）；这里只有 CSV + yaml。

- 每一行 `episode_success` 已经是该 checkpoint 上 **50 条 eval episode 的均值**（`num_eval_episodes: 50`，`eval_every_frames: 2000`）。
- `viz/retrain_process` 的 3 条 rollout 只是进度可视化，**不能**当作业成功率。
- 作业算法文件未改。未覆盖 `viz/`。未重训。

## 主文件（与 viz/retrain_process 同一轮）

| 方法 | 作业 CSV | Logdir run | seed | wandb | 作业门槛 | 正式数字 | 是否达标 |
|---|---|---|---|---|---|---|---|
| PPO | `on_policy.csv` | `run_195512_device=cuda,save_snapshot=true,save_video=false,wandb_group=retrain_process` | 2 | `xgv98hup` | 1M frames 时 ≥ 25% | 1M 附近最后一次 eval：frame=998000 → 18.0% | **否，需补跑** |
| 3A | `off_policy.num_critics=2,utd=1.csv` | `run_195513_device=cuda,save_snapshot=true,save_video=false,wandb_group=retrain_process` | 1 | `14n6qslz` | 100k 前 ≥ 90% | ≤100k 最高：frame=56000 → 64.0%；100k 当时 48.0% | **否，需补跑** |
| 3B | `off_policy.num_critics=10,utd=5.csv` | `run_195513_agent.num_critics=10,device=cuda,save_snapshot=true,save_video=false,utd=5,wandb_group=retrain_process` | 1 | `t4ulxxlu` | 40k 前 ≥ 90% | ≤40k 最高：frame=24000 → 26.0%；40k 当时 0.0% | **否，需补跑** |

### PPO（`CSV files/on_policy.csv`）
- 最后一行：frame=998000，success=18.0%（训练在 1M 前最后一次正式 eval，没有正好 1_000_000 这一行）。
- 全程最高：frame=890000，success=40.0%。
- 首次 ≥25%：frame=0。
- 1M 附近：984k=28%，994k=30%，998k=**18%**。按「最终成功率」**未达 25%**；训练中多次超过 25%，最高 40%。
- 对应 viz：1M 时 2/3、相邻 999998 时 0/3，是 3 条抽样噪声，不能推翻 50-episode 的 18%。

### 3A（`CSV files/off_policy.num_critics=2,utd=1.csv`）
- ≤100k 最高：**64.0%** @ frame=56000。
- 正好 100k：48.0%。
- 首次 ≥90%：frame=250000（在 100k **之后**）。
- **100k 前未达 90%，需要补跑。** viz 里 90k=3/3、105k=1/3 同样只是 3 条抽样。

### 3B（`CSV files/off_policy.num_critics=10,utd=5.csv`）
- ≤40k 最高：**26.0%** @ frame=24000。
- 正好 40k：0.0%。
- 首次 ≥90%：frame=144000（约 144k）。
- **40k 前未达 90%，需要补跑。** 与 viz 30k/45k 都是 0/3 一致：早期确实没达标。

## 结论（这一轮要不要补跑）

| 方法 | 已达标？ | 下一步 |
|---|---|---|
| PPO | 否（最终 18% < 25%；途中最高 40%） | 需要补跑 |
| 3A | 否（100k 前最高 64% < 90%） | 需要补跑 |
| 3B | 否（40k 前最高 26% < 90%） | 需要补跑 |

第一轮 archive 同样没有在门槛内达到作业数字，见下。

## 目录

```
hw2_onlinerl/hw2_submission/
  INDEX.md                 本说明
  inventory.json           机器可读汇总
  CSV files/               作业要求的三份最终文件名（来自主 run）
    on_policy.csv
    off_policy.num_critics=2,utd=1.csv
    off_policy.num_critics=10,utd=5.csv
  runs/ppo|3a|3b/          主 run：eval.csv + hydra_config/overrides + SOURCE.txt
  runs/archive/            更早的完整/近完整 run，便于对照，不是作业主文件
```

Hydra 目录名不用 `.hydra/`，避免被根 `.gitignore` 的 `.hydra/` 规则丢掉。

## 归档 run（不是 viz 那一轮）

| key | Logdir | 正式数字 | 达标 |
|---|---|---|---|
| ppo_early | `run_131159_device=cuda,save_video=false` | 最后 frame=998000 → 22.0%；最高 36.0% @ 922000 | 否 |
| 3a_early | `run_150227_device=cuda,save_snapshot=true,save_video=false` | 门槛内最高 62.0% @ 84000；门槛当时 60.0% | 否 |
| 3b_early | `run_154425_agent.num_critics=10,device=cuda,save_snapshot=true,save_video=false,utd=5` | 门槛内最高 28.0% @ 20000；门槛当时 0.0% | 否 |

跳过：`run_131049_...` PPO 启动失败/中断，没有 `eval.csv`。

## 主 run Hydra 要点

| | PPO | 3A | 3B |
|---|---|---|---|
| agent | `on_policy.PPOAgent` | `off_policy.ACAgent` | `off_policy.ACAgent` |
| seed | 2 | 1 | 1 |
| num_train_frames | 1_000_000 | 300_000 | 300_000 |
| num_eval_episodes | 50 | 50 | 50 |
| eval_every_frames | 2000 | 2000 | 2000 |
| utd / critics | — | 1 / 2 | 5 / 10 |
| save_video | false | false | false |
| wandb_group | retrain_process | retrain_process | retrain_process |

整理时间（服务器）：2026-09-10T10:42:03+08:00
