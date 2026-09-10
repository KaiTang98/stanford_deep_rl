# HW2 轨迹与 RGB 可视化

目录：`hw2_onlinerl/hw2 4/viz/`（已在仓库 `.gitignore` 里）。

## 观测 39 维 / 动作 4 维

| 切片 | 含义 |
|---|---|
| `obs[0:3]` | 夹爪 / 末端 XYZ |
| `obs[3]` | 夹爪开合 `[0,1]`（1=张开） |
| `obs[4:7]` | 锤子 XYZ |
| `obs[7:11]` | 锤子四元数 |
| `obs[11:14]` | 钉子 XYZ |
| `obs[14:18]` | 钉子四元数 |
| `obs[18:36]` | 上一帧的上述 18 维 |
| `obs[36:39]` | 目标 XYZ（包装器关掉了 partial obs） |
| `action[0:3]` | 末端 XYZ 增量（环境 clip 到 `[-1,1]`，再 ×0.01 m） |
| `action[3]` | 夹爪力度 |

## PPO

这轮 PPO 用了 `save_snapshot=false`，**没有** `snapshot.pt`，无法回放终局策略。若以后需要，用 `save_snapshot=true` 再训一次。

## 轨迹图（不依赖渲染）

脚本：`visualize_trajectories.py`

- BC demo（短/中/长）：`demo_demo_4_33.png`、`demo_demo_3_39.png`、`demo_demo_8_46.png`
- 20 条 demo：756 个非 dummy 转移，约 **59%** 的动作至少一维 `|a|>1`（min=-2.54，max=3.21）
- 3A（2 critics, UTD=1）：5 个 eval seed 里 **2/5** 成功
- 3B（10 critics, UTD=5）：同样 **2/5** 成功
- 成功/失败各有一张 representative 图

## RGB 视频

`mujoco_py.sim.render` 在这台机上 **仍然 segfault**（单卡 EGL、去掉 Mesa 混链、device 0–5 都崩；没有 `xvfb`）。

改用 **DeepMind `mujoco` 3.12 只出图**：mujoco_py 只 `step` 存 `qpos`，再交给新渲染器。

- 冒烟：`new_mujoco_smoke_corner.png`
- 脚本：`collect_qpos.py`、`render_qpos_video.py`、`render_smoke_test.py`
- 视频（`corner`，256²，20 fps）：
  - `demo_demo_8_46_replay.mp4`（重放 demo 动作；reset 有噪声，不是逐状态复原）
  - `3A_seed103_success.mp4` / `3A_seed100_fail.mp4`
  - `3B_seed103_success.mp4` / `3B_seed100_fail.mp4`

不要在训练脚本里开 `save_video=true`，第一次 eval 仍会把 mujoco_py 进程打死。
