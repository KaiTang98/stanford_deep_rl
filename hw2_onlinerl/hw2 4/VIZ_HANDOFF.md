# HW2 可视化交接文档（给本地 agent）

> 机器分工：**只在 4090 服务器上跑** MuJoCo / 训练 / 录像；本地 agent 只读本文档，可改**独立** debug 脚本，不要改作业算法三文件，也不要在本地装 Meta-World。  
> 课程禁止用生成式模型写 actor-critic 公式；本文档只描述**非算法**可视化与已有补丁。

---

## 1. 目的与机器

- **问题**：`viz/retrain_process/expert/index.md` 写 `demo_4_33 … success=True`，但 `demo_4_33.mp4` 看起来像失败。
- **结论（见第 4、7 节验证）**：原始 `demos/demo_4_33.npz` 是成功专家轨迹；mp4 不是原轨迹，而是「随机 `reset` + 开环重放 `action`」得到的另一条轨迹。标签来自原 npz，**不是**回放是否成功。
- **环境**：conda `cs224r-hw2-local`；`mujoco_py` 的 `sim.render` 在本机 **segfault**；RGB 用 DeepMind **`mujoco` 3.x** 从保存的 `qpos` 离屏渲染。
- **激活（服务器，勿写进 `~/.bashrc`）**：

```bash
conda activate cs224r-hw2-local
unset LD_LIBRARY_PATH CPATH LIBRARY_PATH
source ~/.local/hw2-sysdeps/env.sh
cd ~/ws/stanford_deep_rl/hw2_onlinerl/hw2\ 4
```

---

## 2. 从「开始要求渲染」以来改过什么

### 2.1 作业算法（未改 / 不要动）

| 文件 | 说明 |
|------|------|
| `on_policy.py` | GAE + PPO（学生代码） |
| `off_policy.py` | BC / critic / actor（学生代码） |
| `gridworld_q_learning.py` | P1 |

### 2.2 训练入口（已改，非公式）

[`train_on_policy.py`](train_on_policy.py)、[`train_off_policy.py`](train_off_policy.py)：

1. **Demo 路径**：不用 Modal 的 `/root/demos`，改为仓库内 `Path(__file__).parent / "demos"`，再 `copy_tree` 到 Hydra `work_dir`。
2. **Eval 录像钩子**：每轮 eval 的第 0 条 episode 结束后 `video_recorder.save(f'{global_frame}.mp4')`。训练命令仍用 **`save_video=false`**（否则第一次 `sim.render` 会崩）。
3. **进度可视化（可选）**：若环境变量 `HW2_PROGRESS_METHOD` 为 `ppo` / `3a` / `3b`，则挂上 [`viz_progress.py`](viz_progress.py) 的 `ProgressDumper`：在 0%、每 5%、接近结束时各 dump 3 条随机 eval（png + qpos + mp4）。

### 2.3 环境包装（已改，非公式）

[`mw.py`](mw.py)：`render()` 用 `self._env.sim.render(..., camera_name='corner')` 再竖直翻转。仅在 `save_video=true` 时会走这条路径；**当前训练不要开**。

### 2.4 新增可视化脚本（可动，不影响训练公式）

| 脚本 | 作用 |
|------|------|
| `visualize_trajectories.py` | 39 维 obs / 4 维 action 切片；画 3D + 动作时序图 |
| `render_smoke_test.py` | 独立进程测 `mujoco_py` 离屏（本机失败） |
| `collect_qpos.py` | mujoco_py 只 `step`，存 `qpos` |
| `render_qpos_video.py` | **mujoco 3.x** 把 `qpos.npz` 渲成 mp4（不要 import metaworld） |
| `viz_progress.py` | 训练每 5% dump 3 条 rollout |
| `dump_expert_progress.py` | 专家 demo：原 obs 画 png；**开环重放 action** 存 qpos 再渲 mp4 |
| `debug_expert_demo.py` | **只读 debug**：对照原 npz vs 开环回放（见第 6 节） |

### 2.5 产物目录（gitignore）

- `viz/`：早期静态可视化
- `viz/retrain_process/`：重训过程可视化
  - `expert/`：短/中/长三条 demo（png + mp4 + qpos）
  - `ppo/`、`3a/`、`3b/`：各含 `expert/` 副本 + `pctXXX_frameYYYYYYY/`
- `viz/retrain_process/expert_debug/`：本 debug 脚本输出

仓库根 `.gitignore` 含 `hw2_onlinerl/hw2 4/viz`。

---

## 3. 两条管线（不要混）

```text
专家视频 (dump_expert_progress.py)
  原 npz.obs ──► png / index「success」(原标签)
  原 npz.action ──► env.reset(随机手) ──► 开环 step ──► qpos ──► mujoco3 mp4
  保存的 qpos.npz 里的 success 字段 = 原标签，不是回放成败

策略进度视频 (viz_progress.py)
  agent.act + env.step ──► 当时真实 qpos + last reward ──► png/mp4
  标签与画面应对得上
```

**`demo_4_33.png`**：用原始 `observation`，应像成功。  
**`demo_4_33.mp4`**：用开环回放的 `qpos`，常失败。

---

## 4. `demo_4_33` 现象与假设

| 来源 | 内容 |
|------|------|
| 原 `demos/demo_4_33.npz` | `T=33`；reward 仅最后一步 `1.0`；钉子 y：`0.64 → ~0.724`（目标 `0.74`）→ **成功** |
| `index.md` | `success=True` ← 原 npz，不是 mp4 |
| `mw.reset` | 手位 `hand_init + 0.03*N(0,I)`，再空动作 10 步 |
| 动作 | 约一半步至少一维 `|a|>1`，环境 clip 到 `[-1,1]` |
| 视频 | 在**另一初始状态**开环重放 → 可以失败，**不是 demo 坏了，也不是 mujoco3 渲错标签** |

---

## 5. 本地 agent / 服务器分工

**本地 agent 可以：**

- 读 `VIZ_HANDOFF.md`、`viz/retrain_process/**/index*`、脚本源码
- 改/写**独立**脚本（如 `debug_expert_demo.py`），不碰公式与训练逻辑

**本地 agent 不要：**

- 改 `on_policy.py` / `off_policy.py` / `gridworld_q_learning.py`
- 为这次 debug 再改 `train_*.py` / `mw.py`（除非用户明确要求修专家重渲）
- 在 Mac 上装 MuJoCo / 跑 `train_*.py`

**只在服务器跑：**

```bash
# 见第 1 节激活后
CUDA_VISIBLE_DEVICES=0 python debug_expert_demo.py
# 结果在 viz/retrain_process/expert_debug/
```

若以后要「专家 mp4 也是真成功轨迹」，应改 `dump_expert_progress.py`（仍非作业公式）：用 obs 做运动学/固定初始，并分开写 `demo_success` / `replay_success`。**本次 debug 不做重渲。**

---

## 6. Debug 脚本用法

```bash
CUDA_VISIBLE_DEVICES=0 python debug_expert_demo.py
# 可选
CUDA_VISIBLE_DEVICES=0 python debug_expert_demo.py --demo demos/demo_4_33.npz --seed 0
```

输出目录：`viz/retrain_process/expert_debug/`

- `report.md` / `report.json`：原轨迹 vs 开环回放对照
- `replay_last_frame.png`：已存 qpos 最后一帧（mujoco 3）
- `original_trajectory_plot.png`：原 obs 轨迹图（对照用）

---

## 7. 服务器验证结果

脚本：`CUDA_VISIBLE_DEVICES=0 python debug_expert_demo.py --seed 0`  
产物：[`viz/retrain_process/expert_debug/report.md`](viz/retrain_process/expert_debug/report.md)

| 项 | 数值 |
|----|------|
| 原 npz `demo_success` | **True**（T=33，reward 仅最后一步 1.0） |
| 钉子 y | `0.640 → 0.724`（目标 `0.740`），nail→goal 距离 `0.100 → 0.016` |
| 动作 `|a|>1` 比例 | **51.5%**，范围约 `[-1.60, 2.27]` |
| 开环回放 `replay_success`（seed=0） | **False**（last_reward=0，钉子仍停在 y=0.64） |
| 原 obs vs 回放 obs RMSE | hand **0.051**，hammer **0.061**，nail **0.012**，full **0.045** |

**结论**：demo 本身没问题；`index`/`png` 标成功是对的；专家 **mp4** 是另一条开环回放，失败正常。不是 mujoco 3 渲染把成功画成失败，而是**回放状态本来就没成功**。

详细对照与最后一帧图见 `viz/retrain_process/expert_debug/`。
