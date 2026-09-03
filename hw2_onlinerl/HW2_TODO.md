# CS224R HW2 自学进度清单

> **课程**: CS224R Deep Reinforcement Learning — Online Reinforcement Learning  
> **作业截止（官方）**: 2026/4/24  
> **你的状态**: 自学，不提交 Gradescope；完成后请 AI 多模型 review 打分  
> **官方提醒**: 课程禁止用生成式模型**写** actor-critic 代码；本清单只规划进度，实现请自己写。长训练建议尽早开跑（PPO 到 1M steps、UTD=5 大约 2 小时 / 50k steps）

---

## 使用说明

- 按顺序从上到下做，每完成一项把 `[ ]` 改成 `[x]`
- 实验结果填在对应「记录区」，截图/数字最后汇总进 `[CS224R_2026_Homework_2.tex](CS224R_2026_Homework_2.tex)`
- 作业 PDF: `[CS224R_2026_Homework_2.pdf](CS224R_2026_Homework_2.pdf)`
- 代码目录: `[hw2 4/](hw2%204/)`（starter 解压后的文件夹名）
- Modal 计算指南: `[CS224R_compute_guide.pdf](CS224R_compute_guide.pdf)`
- 算法上学生**只应改**这三份:
  - `gridworld_q_learning.py`
  - `on_policy.py`
  - `off_policy.py`
- 在自己的 Linux GPU 服务器上训练时，还需要两处**非算法**小补丁（见 Phase 0）：demo 路径 `/root/demos` → 仓库内 `demos/`，以及 `device=cuda`
- **算力原则（自学）**: Mac 写代码 + 跑 P1/单测；**6 卡 4090 服务器训练**；不要为这门作业装 Mac 版 MuJoCo；5090 / Modal 只当备选

### 机器怎么分工

| 机器 | 用来做什么 | 不要做什么 |
| --- | --- | --- |
| Mac（本机） | 读 PDF、写三个算法文件、跑 Gridworld、跑 GAE 单测、写 LaTeX、看 WandB | **不要**装 Meta-World / `mujoco_py`，不要跑 `train_*.py` |
| **6×4090 服务器（主训练机）** | P2 PPO 1M、P3 off-policy 两条曲线 | 不要 6 卡并行一个 job（代码是单进程单卡） |
| 5090 服务器 | 仅当 4090 实在装不上环境时再考虑 | 不要直接套 `conda_env_local.yml`（里面是 CUDA 11.5，5090 跑不了） |
| Modal | 环境装了 1–2 小时仍失败时的退路 | 自学有 GPU 就不必一上来用 |

为什么选 4090 而不是 5090 / Modal：

- 网络是很小的 MLP，瓶颈在 MuJoCo 仿真，5090 几乎没有收益
- 课程栈是 `mujoco_py` + 偏旧的依赖；4090（sm_89）兼容性最好；5090（Blackwell）要新 PyTorch/CUDA 12.8+，装环境更容易卡
- 6 卡的价值是 **两条训练同时跑**：GPU0 挂 PPO（最长），GPU1 挂 off-policy
- 官方 Modal 镜像是给没 GPU 的学生用的；你已经有机器，少一层账号/credits/镜像构建

### 常用命令

```bash
# ===== Mac =====
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cs224r          # HW1 那个环境即可；P1 + 单测够用
cd "hw2_onlinerl/hw2 4"
python gridworld_q_learning.py
python tests/test_on_policy.py

# ===== 4090 服务器（tmux 里跑）=====
conda activate cs224r-hw2-local
cd ~/cs224r-hw2/hw2_onlinerl/hw2\ 4     # 路径按你 clone/rsync 的位置改

# P2：1M frames，单独占一张卡
CUDA_VISIBLE_DEVICES=0 python train_on_policy.py device=cuda

# P3A：2 critics, UTD=1（作业要 100k 前 ≥90%，默认 config 会跑到 300k）
CUDA_VISIBLE_DEVICES=1 python train_off_policy.py device=cuda

# P3B：10 critics, UTD=5（作业要 40k 前 ≥90%）
CUDA_VISIBLE_DEVICES=2 python train_off_policy.py device=cuda agent.num_critics=10 utd=5
```

---

## Phase 0: 环境、账号与预习

### 0.1 Mac：只做开发，不训练 P2/P3

本机 **可以**跑的：`gridworld_q_learning.py`（纯 numpy）、`tests/test_on_policy.py`（只要有 torch）、写代码、编译 PDF。  
本机 **不要**跑的：`train_on_policy.py` / `train_off_policy.py`。它们依赖 Linux + `mujoco_py` + EGL，和 HW1 的 gymnasium/pygame 不是同一套。

- [ ] 本机继续用 HW1 的 `cs224r` 环境即可（有 numpy / torch 就行）
- [ ] 注册 [Weights & Biases](https://wandb.ai/site)，本机 `wandb login`（只为了浏览器看曲线；训练在服务器上 login）
- [ ] **不要**在 Mac 上 `conda env create -f conda_env_local.yml`

### 0.2 4090 服务器：一次性训练环境（这是唯一值得花时间的安装）

目标：在 **其中一张 4090** 上先 `import metaworld, mujoco_py, torch; torch.cuda.is_available()` 为 True。6 张卡共用这一个 conda 环境。

**不要**原样执行 `conda_env_local.yml`：里面的 `cudatoolkit=11.5` 过时，和现在的 4090 驱动经常对不上。用「新 PyTorch + 课程 pin 住的 Meta-World」更省事。

```bash
# 系统依赖（Ubuntu；包名按发行版微调）
sudo apt-get update
sudo apt-get install -y libglew-dev patchelf libosmesa6-dev \
    libgl1-mesa-glx libglfw3 libglew2.2 libegl1-mesa build-essential wget

# MuJoCo 2.1.0（mujoco_py 需要这个，不是 conda 里的 mujoco）
mkdir -p ~/.mujoco && cd ~/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Python 环境：按你机器已有的 CUDA 驱动选 cu121 / cu124 即可
conda create -n cs224r-hw2-local python=3.10 -y
conda activate cs224r-hw2-local
pip install torch torchvision torchaudio  # 4090：cu121/cu124 都行
pip install "numpy==1.24.3" "Cython==0.29.33" "gym==0.26.2" \
    "hydra-core==1.1.0" hydra-submitit-launcher==1.1.5 \
    wandb dm_control termcolor imageio imageio-ffmpeg \
    opencv-python pandas matplotlib scikit-learn
pip install "mujoco_py==2.1.2.14"   # 第一次会本地编译，失败多半是缺上面的 apt 包
pip install "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb"

python - <<'PY'
import torch, mujoco_py, metaworld
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("mujoco_py", mujoco_py.__file__)
PY
```

- [ ] 上述 smoke test 打印出 GPU 名 + mujoco_py 路径
- [ ] 服务器 `wandb login`（训练脚本默认 `use_wandb: true`）
- [ ] 把作业目录 rsync/git clone 到服务器（**不要**在 OneDrive 网络盘上直接训练）

如果 `mujoco_py` 编译死磕超过 ~1–2 小时：停，改用 Modal（见 0.4），不要再换 5090 硬装同一套旧栈。

### 0.3 服务器训练前必须做的两处非算法补丁

Starter 是按 Modal 写的：`train_*.py` 会 `copy_tree("/root/demos/", ...)`，config 里 `device: cpu`。在 4090 上不改这两处会直接报错或慢到不可用。

- [ ] `train_on_policy.py` / `train_off_policy.py`：demo 目录改为仓库内路径，例如

```python
from pathlib import Path
_demo_src = Path("/root/demos")
if not _demo_src.exists():
    _demo_src = Path(__file__).resolve().parent / "demos"
copy_tree(str(_demo_src), str(self.work_dir / "demos"))
# off-policy 里 buffer 那份 copy 同样改成 _demo_src
```

- [ ] 启动时覆盖设备（不必改 yaml）：`python train_xxx.py device=cuda`
- [ ] （建议）关视频，少一个渲染坑：`save_video=false`
- [ ] 长任务放 `tmux` / `screen`；用 `CUDA_VISIBLE_DEVICES` 绑单卡，不要试图改代码做多卡

### 0.4 备选：5090 / Modal（默认跳过）

- **5090**：只有 4090 完全没权限时才用。不要用 `conda_env_local.yml`。PyTorch 需 CUDA 12.8+（Blackwell）。`mujoco_py` 编译风险比 4090 更大，收益几乎为零。
- **Modal**：环境已经打好，代价是镜像、secret、credits。若走这条：本机另装 `conda_env_modal.yml`，`modal secret create wandb-secret WANDB_API_KEY=... --force`，然后 `modal run --detach modal_on_policy.py`。UTD=5 实验仍按 PDF 改 `modal_off_policy.py` 的 argv。

### 0.5 通读材料

- [ ] 通读 `[README.md](hw2%204/README.md)`（三个 baseline、Hydra 入口、config）
- [ ] 通读作业 PDF Overview + Setup + 提交格式
- [ ] 通读只读文件（理解 pipeline，**不要改算法逻辑**）:
  - [ ] `train_on_policy.py` — PPO 训练循环：BC pretrain → 收集 rollout → `agent.update`
  - [ ] `train_off_policy.py` — AC 训练循环：BC pretrain → replay 更新 critic（`utd` 次）→ actor / 间歇 BC
  - [ ] `mw.py` — Meta-World `hammer-v2`：sparse reward（成功=1）、`action_repeat=2`、episode 最长 50
  - [ ] `replay_buffer.py` — demo / replay；off-policy 默认 **n-step=3**（batch 里的 `reward`/`discount` 已经是 n-step 聚合）
  - [ ] `utils.py` — `TruncatedNormal`、`soft_update_params`、`to_torch`
  - [ ] `logger.py` — WandB + CSV；报告要截的是 `eval/episode_success`
  - [ ] `cfgs/on_policy_config.yaml`、`cfgs/off_policy_config.yaml`
  - [ ] `modal_on_policy.py`、`modal_off_policy.py`、`modal_gridworld_q_learning.py`
- [ ] 编译一次报告模板确认 LaTeX 可用:
  ```bash
  cd hw2_onlinerl
  latexmk -pdf CS224R_2026_Homework_2.tex
  ```

**关键概念速记**

| 项目 | 值 |
| --- | --- |
| 任务 | Meta-World `hammer-v2`（Sawyer 4-DoF，连续动作）+ 一个 5×4 Gridworld |
| 观测 | 环境状态（夹爪/锤子 pose 等）；off-policy 文本说含最近两步 |
| 动作 | 连续，TruncatedNormal 截到 **[-1, 1]** |
| 奖励 | **sparse**：完成任务才 1.0，中间步 0 |
| Demo | `demos/` 里 20 条成功轨迹，BC pretrain |
| PPO 训练长度 | `num_train_frames=1_000_000`，目标 **≥ 25% success** |
| Off-policy 默认 | 2 critics，UTD=1，`num_train_frames=300_000`，目标 **100k steps 前 ≥ 90%** |
| Off-policy UTD 实验 | 10 critics，UTD=5，目标 **40k steps 前 ≥ 90%** |
| 评估 | 每 2000 frames、50 eval episodes；曲线看 WandB `eval/episode_success` |

---

## Phase 1: Problem 1 — Gridworld Q-Learning（3 分）

文件: `[gridworld_q_learning.py](hw2%204/gridworld_q_learning.py)`

地图：5×4，start `(0,0)`，Goal 2 `(4,0)`（约 4 步），Goal 1 `(4,3)`（约 7 步）。动作 `{left, right, down, up}`；撞墙原地不动但仍扣 `r_step`。

更新公式：

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \big[ r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t) \big]
\]

默认超参（代码里已写好）：`episodes=5000`，`horizon=20`，`α=0.2`，`γ=0.98`，`ε: 0.4 → 0.02` 线性衰减。

### 1.1 实现代码

- [ ] `choose_action` — ε-greedy：以 ε 均匀随机，否则 greedy；**平局任意打破**
- [ ] `greedy_action` — 代码里也是 `YOUR CODE HERE`（PDF 没单独列，但 `rollout_policy` 会调用；实现 `argmax_a Q(s,a)`）
- [ ] `train_q_learning` 内层循环，每步:
  1. ε-greedy 选动作
  2. `env.step` 得到 `(s', r, done)`
  3. Q-learning 更新
  4. `done` 则结束该 episode

**自测**: 实现后应不再是 `pass`；跑起来三个 scenario 的 `observed` 应分别接近 `goal_1` / `goal_2` / `timeout`（代码里的 `expected_outcome`）

### 1.2 跑实验

- [ ] 本机:
  ```bash
  python gridworld_q_learning.py
  ```
- [ ] （不必上 GPU / Modal）

**记录区 — Problem 1 结果**

| Scenario | \(r_{step}\) | \(R_1\) | \(R_2\) | 是否到达 goal | 哪个 goal | 轨迹 / 总回报 | 一句话原因 |
| -------- | ------------ | ------- | ------- | ------------- | --------- | ------------- | ---------- |
| 1 | -1 | 10 | 5 |  |  |  |  |
| 2 | -2 | 10 | 5 |  |  |  |  |
| 3 | +1 | 1 | 1 |  |  |  |  |

提示（写报告前先自己想，不要直接当答案抄）:

- Scenario 1：每步惩罚轻，走远路拿 Goal 1 的净回报可能更高
- Scenario 2：每步惩罚加倍，近的 Goal 2 更划算
- Scenario 3：每步 **正** 奖励，agent 可能故意耗满 horizon、不进 terminal

### 1.3 写报告

- [ ] 每个 scenario：是否到达 goal、到达哪一个
- [ ] 每个 scenario 一句解释「为什么学到这条轨迹」

---

## Phase 2: Problem 2 — On-policy PPO（3 分）

文件: `[on_policy.py](hw2%204/on_policy.py)`  
入口: 服务器上 `python train_on_policy.py device=cuda`（官方文档写的是 `modal_on_policy.py`）  
**不要改**除 `on_policy.py` 以外的算法文件。demo 路径补丁见 Phase 0.3。

组件（已实现，只需填更新）:

- Actor: TruncatedNormal，动作 [-1, 1]
- Critic: \(V_\phi(s)\)（只看状态）
- Frozen reference actor: BC 结束后 `set_reference_policy()`，PPO 里 reverse-KL 正则

### 2.1 实现代码

- [ ] `compute_gae`：从 \(t=T-1,\ldots,0\) 反传

\[
\delta_t = r_t + \gamma (1-d_t) V(s_{t+1}) - V(s_t),\quad
\hat A_t = \delta_t + \gamma\lambda (1-d_t)\hat A_{t+1}
\]

\[
\hat R_t = \hat A_t + V(s_t)
\]

  - `self.gae_lambda`、`self.gamma` 已有
  - **done 时不要 bootstrap**（用 `dones`）
  - 函数还传入 `discounts`：训练时它已经是 `γ * env_discount`；`tests/test_on_policy.py` 里 `discounts` 恒为 0.99，**必须再乘 `(1-done)`** 才能过 GAE 单测

- [ ] `update` 里第一处 `YOUR CODE HERE`（`torch.no_grad()` 内）:
  - 用 critic 算 `values`、`next_values`
  - 调 `compute_gae` 得到 `advantages_all`、`returns_all`
  - 这些是 target，不要让梯度流过

- [ ] `update` 里第二处：PPO-Clip
  1. `ratio` \(\rho_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})\)（用 log 差，不要直接除概率）
  2. `policy_loss`（**最小化**，所以带负号）:

\[
L^{\mathrm{CLIP}}(\theta) = -\frac{1}{B}\sum_t \min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat A_t\big)
\]

  - `ε = self.clip_eps`；后面 reverse-KL 会用到你定义的 `ratio`，名字不要改

**本机单测（不需要 Modal / MuJoCo）**:

```bash
python tests/test_on_policy.py
```

- [ ] `test_compute_gae_matches_manual_recursion` 通过
- [ ] （可选）把 clip 实现贴进测试文件里的第二段，跑 `test_clipped_surrogate_objective`

### 2.2 跑实验（4090，约 1M env frames）

默认关键超参（`cfgs/on_policy_config.yaml`）: `rollout_length=4096`，`batch_size=64`，`pretrain_steps=10000`，`clip_eps=0.1`，`ppo_epochs=3`，`gae_lambda=0.99`，`gamma=0.99`，`hidden_dim=64`，`lr=3e-4`。  
官方 Modal 用 A10；你这边 **一张 4090 足够**。墙钟大约 1–3 小时量级（仿真是瓶颈，别指望 5090 快很多）。

- [ ] 4090 上 `tmux` 里启动（单卡）:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python train_on_policy.py device=cuda save_video=false
  ```
- [ ] 打开 WandB，确认 `eval/episode_success` 在刷
- [ ] 跑到 **1 million steps**；成功率应 **≥ 25%**

**记录区 — Problem 2 结果**

| 项目 | 值 |
| ---- | -- |
| WandB run URL |  |
| 1M steps 时 eval success |  |
| 是否 ≥ 25% |  |
| 截图路径 | `figures/ppo_eval_success.png` |

- [ ] 保存 WandB `eval/episode_success` 截图（横轴到 1M）
- [ ] 下载 CSV: Charts → `eval/episode_success` → 右上角三点 → CSV  
  保存为 `on_policy.csv`

### 2.3 写报告

- [ ] 插入 `eval/episode_success` 截图（到 1M steps）

---

## Phase 3: Problem 3 — Off-policy Actor-Critic（4 分）

文件: `[off_policy.py](hw2%204/off_policy.py)`  
入口: 服务器上 `python train_off_policy.py device=cuda`（官方文档写的是 `modal_off_policy.py`，自学走本地脚本即可）

组件（已搭建）:

- Actor \(\pi_\theta(a|s)\)
- Critic ensemble \(Q_{\phi_i}(s,a)\)，\(i=1\ldots N\)
- Target critics \(\bar Q_{\phi_i}\)，EMA 更新

训练循环要点（已写在 `train_off_policy.py`，不用改）:

1. BC pretrain **2000** steps（写死在 train script）
2. 每环境步：先 `update_critic` **`utd` 次**
3. warmup（2000 frames）之后再 `update_actor`；每 `bc_freq=2` 步再 BC 一次

### 3.1 实现代码

- [ ] `bc`：监督行为克隆

\[
L_{\pi}(s_t,a_t) = -\log\pi_\theta(a_t|s_t)
\]

  对 batch 取 mean，反传 **只更新 actor**。可参考 `on_policy.py` 里已经写好的 `PPOAgent.bc`。

- [ ] `update_critic`（Bellman）:
  1. 从当前 policy 采样 \(a_{t+1} \sim \pi_\theta(s_{t+1})\)（可用 `dist.sample(clip=self.stddev_clip)`）
  2. **无放回**抽两个 target critic：`random.sample(list, 2)`，取 **min**
  3. Target（`sg` = stop gradient）:

\[
y = r_t + \gamma_{\text{batch}} \min\{\bar Q_i(s_{t+1},a_{t+1}),\bar Q_j(s_{t+1},a_{t+1})\}
\]

     这里的 `reward` / `discount` **直接用 batch 里的**（replay 已做 n-step=3），不要自己再乘一遍 `0.99³`
  4. **所有 N 个** critic 都对同一个 `y` 做 MSE，再求和/平均后反传
  5. `utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)`  
     即 \(\bar Q \leftarrow (1-\rho)\bar Q + \rho Q\)，`ρ = critic_target_tau = 0.005`

- [ ] `update_actor`:
  1. \(a' \sim \pi_\theta(\cdot|s)\)
  2. 最大化所有 critic 的平均 Q（实现成 minimize）:

\[
L_\pi = -\frac{1}{N}\sum_{i=1}^{N} Q_{\phi_i}(s, a')
\]

  3. 梯度只走 actor，**不要**更新 critic（对 Q 的调用保持 `torch.no_grad()` 或 detach）

PDF 里的常见 bug 对照:

- critic loss 一直平 → 没真正 `optimizer.step`
- critic loss 爆炸 → `y` 和 `Q` 的 shape / broadcast 错（都应是 `[batch, 1]` 一类）
- 只更新了被采样的那两个 critic → 必须 **N 个全更新**；随机采样只用于构造 target

### 3.2 跑实验 A：2 critics，UTD=1（1 分）

默认 config：`utd=1`，`agent.num_critics=2`，`batch_size=256`，`nstep=3`，`lr=1e-4`，`hidden_dim=256`。  
作业只要看到 **100k steps 前 ≥ 90%**；默认 `num_train_frames=300000` 可以继续跑，曲线更好看。可与 PPO 同时占另一张卡。

- [ ] 启动:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python train_off_policy.py device=cuda save_video=false
  ```
- [ ] 看 WandB 到 **100k steps**；成功率应 **≥ 90%**

**记录区 — Problem 3A（UTD=1）**

| 项目 | 值 |
| ---- | -- |
| WandB run URL |  |
| 100k 时 eval success |  |
| 首次 ≥ 90% 的 step |  |
| 截图路径 | `figures/offpolicy_utd1_eval_success.png` |

- [ ] 保存截图（到 100k）
- [ ] 下载 CSV，存为 `off_policy.num_critics=2,utd=1.csv`

### 3.3 跑实验 B：10 critics，UTD=5（1 分）

- [ ] **不必改** `modal_off_policy.py`，Hydra 命令行覆盖即可:
  ```bash
  CUDA_VISIBLE_DEVICES=2 python train_off_policy.py device=cuda save_video=false agent.num_critics=10 utd=5
  ```
  若走 Modal 退路，才按 PDF 去改 `modal_off_policy.py` 的 argv。
- [ ] 看 WandB 到 **40k steps**；成功率应 **≥ 90%**  
  预期：更高 UTD → critic 每步更新更多 → 通常 **更 sample-efficient**，但 wall-clock 更慢（官方约 **2 小时 / 50k steps**；4090 同量级）

**记录区 — Problem 3B（UTD=5）**

| 项目 | 值 |
| ---- | -- |
| WandB run URL |  |
| 40k 时 eval success |  |
| 首次 ≥ 90% 的 step |  |
| 截图路径 | `figures/offpolicy_utd5_eval_success.png` |
| 相对 3A 的变化（一句） |  |

- [ ] 保存截图（到 40k）
- [ ] 下载 CSV，存为 `off_policy.num_critics=10,utd=5.csv`
- [ ] 写一句解释：为什么 UTD=5 + 更多 critic 会更快到 90%

### 3.4 对比 PPO vs Actor-Critic（2 分）

对比 **Problem 2 的 PPO 曲线** 和 **Problem 3A（2 critics, UTD=1）** 的 `eval/episode_success`。

- [ ] 3–5 句，至少写出 **两个具体差异**（sample efficiency、最终性能）
- [ ] 每个差异都接到算法性质，例如:
  - off-policy + replay / n-step / UTD vs on-policy 用完即弃的 rollout
  - Q-ensemble + clipped min target vs GAE + \(V(s)\)
  - PPO clip / reverse-KL 保守更新 vs actor 直接 max Q
  - 最终成功率：作业预期 off-policy 可接近 100%，PPO 过 25% 即可

**记录区 — 对比草稿**

- 差异 1:
- 差异 2:
- （可选）差异 3:

---

## Phase 4: 收尾

### 4.1 报告 PDF

- [ ] 填写页眉：Name、Collaborators（自学可写自己的）
- [ ] 检查所有红色占位已换成结果/截图/文字
- [ ] 三张 WandB 图轴范围正确：PPO→1M，off-policy A→100k，off-policy B→40k
- [ ] 编译:
  ```bash
  cd hw2_onlinerl
  latexmk -pdf CS224R_2026_Homework_2.tex
  ```
- [ ] 通读最终 PDF

### 4.2 代码与 CSV 归档（自学备份；官方 zip 格式如下）

```text
submit.zip
├── gridworld_q_learning.py
├── on_policy.py
├── off_policy.py
└── CSV files
    ├── on_policy.csv
    ├── off_policy.num_critics=2,utd=1.csv
    └── off_policy.num_critics=10,utd=5.csv
```

- [ ] 整理 `hw2_submission/`（或同等目录）
- [ ] 确认三份 CSV 来自**最终**那三次 run，不是调试 run

---

## Phase 5: 完成后 — 多模型 Review 打分

> 你做完后回来找我，我会开**多个 agent、使用不同模型**分别 review，再汇总打分。

### 你需要提供

1. 代码：`gridworld_q_learning.py`，`on_policy.py`，`off_policy.py`
2. 三份 WandB CSV + 三张 `eval/episode_success` 截图
3. 编译好的 PDF 报告
4. （可选）Modal / WandB run 链接

### Review 维度（按官方 10 分）

| 维度 | 分值 | 检查内容 |
| --- | --- | --- |
| P1 三个 scenario | 3 | 到达哪个 goal / timeout；解释是否对应步惩罚与折扣 |
| P2 代码 | （含在 3 分实验里） | GAE 反传、done mask、PPO-Clip 的 `ratio` + `policy_loss` |
| P2 实验 | 3 | 1M steps 曲线；success ≥ 25% |
| P3 代码 | （含在实验里） | BC、min-of-2-target、全体 critic MSE、soft update、actor max-Q |
| P3A 实验 | 1 | 100k 前 ≥ 90% |
| P3B UTD | 1 | 40k 前 ≥ 90%；一句话解释 sample efficiency |
| P3 对比 | 2 | PPO vs AC，至少两个具体差异并接到算法性质 |
| **合计** | **10** | |

### 触发方式

完成后直接说：

> 「HW2 做完了，帮我打分 review」

并附上代码路径、PDF、CSV/截图位置即可。

---

## 推荐实现顺序

```
Mac: choose_action → greedy_action → Q-learning 内环 → 本机跑 3 个 scenario → 写 P1
    ↓
Mac: compute_gae → 单测 → PPO update（values/GAE + clip loss）
    ↓  同时：4090 GPU0 挂上 PPO 1M（最长，先丢着）
Mac: bc → update_critic → update_actor
    ↓  4090 GPU1: off-policy UTD=1
    ↓  4090 GPU2（或等 GPU1 结束）: UTD=5
对比曲线 → 写报告 → 编译 PDF → 找我 review
```

**并行建议**: PPO 1M 最慢，Mac 上 GAE 单测一过就上 GPU0；写 P3 代码时 PPO 在跑。6 卡不要用来拆一个 job，只用来并排跑独立实验。

---

## 进度总览

| Phase | 内容 | 状态 |
| ----- | ---- | ---- |
| 0 | 4090 环境 + WandB + 预习 | ⬜ |
| 1 | Gridworld Q-learning | ⬜ |
| 2 | PPO（GAE + clip） | ⬜ |
| 3 | Off-policy AC + UTD + 对比 | ⬜ |
| 4 | 报告 + 归档 | ⬜ |
| 5 | 多模型 Review | ⬜ |

---

*最后更新: 2026-09-03*
