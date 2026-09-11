# CS224R HW3 自学进度清单

> **课程**: CS224R Deep Reinforcement Learning — Offline Reinforcement Learning  
> **作业截止（官方）**: 2026/5/8  
> **你的状态**: 自学，不提交 Gradescope；完成后请 AI 多模型 review 打分  
> **官方提醒**: 课程禁止用生成式模型**写** AWAC / IQL 代码；本清单只规划进度，实现请自己写。长训练建议尽早开跑（官方 T4 估 umaze ~1h / seed、medium ~1.5h / seed；4090 离线更新通常更快）

---

## 使用说明

- 按顺序从上到下做，每完成一项把 `[ ]` 改成 `[x]`
- 实验结果填在对应「记录区」，截图/数字最后汇总进 `[CS224R_2026_Homework_3.tex](CS224R_2026_Homework_3.tex)`
- 作业 PDF: `[CS224R_2026_Homework_3.pdf](CS224R_2026_Homework_3.pdf)`
- 代码目录: `[hw3/](hw3/)`
- 算法上学生**只应改**这五份:
  - `cs224r/policies/MLP_policy.py` — AWAC / IQL 共用 actor（AWR loss）
  - `cs224r/critics/awac_critic.py`
  - `cs224r/agents/awac_agent.py`
  - `cs224r/critics/iql_critic.py`
  - `cs224r/agents/iql_agent.py`
- BC / Filtered BC、trainer、PointMass 环境都是现成的，**不要改算法逻辑**
- **算力原则（自学）**: Mac 写代码；**6 卡 4090 服务器训练**；Modal 只当 D4RL 装崩时的退路（自学没有课内 credits；Starter 绑卡每月送 $30，整份作业约 $12–25，通常够，但没必要一上来买）

### 机器怎么分工

| 机器 | 用来做什么 | 不要做什么 |
| ---- | ---------- | ---------- |
| Mac（本机） | 读 PDF、填五个 TODO 文件、写 LaTeX、看 WandB、编译报告 | **不要**在 Mac 上跑 AntMaze / D4RL；不要给 HW3 装 Mac 版 MuJoCo |
| **6×4090 服务器（主训练机）** | 全部实验：AWAC / IQL AntMaze + PointMass stitching | **不要**复用 `cs224r-hw2-local`；不要 6 卡并行一个 job（代码是单进程单卡） |
| Modal | 环境装了 1–2 小时仍失败时的退路 | 自学没有课内兑换码；不要为这门课单独充 credits |

为什么选 4090 而不是 Modal：

- 训练是离线的：循环里几乎不 step 环境，就是 replay + MLP，4090 比官方 T4 快
- HW2 已经在这台机器上装过 MuJoCo 2.1.0 + `mujoco_py` 的系统依赖，HW3 只差 **新 conda + gym 0.23 + D4RL**
- 6 卡的价值是 **3 个 seed 并排**：GPU0/1/2 各挂一个 `--seed`
- 官方 Modal 是给没 GPU 的选课学生用的；你已经有机器，少一层账号 / 绑卡 / 镜像构建

本地 argparse 是下划线 `--env_name`；Modal CLI 才是连字符 `--env-name`。服务器一律走 `cs224r/scripts/run_algo.py`。

### 常用命令

```bash
# ===== Mac =====
# 写代码、编译 PDF 即可；不必为 HW3 新建 conda
cd hw3_offlinerl
latexmk -pdf CS224R_2026_Homework_3.tex

# ===== 4090 服务器（tmux 里跑）=====
conda activate cs224r-hw3
cd ~/cs224r-hw3/hw3_offlinerl/hw3     # 路径按你 clone/rsync 的位置改
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1

# 单 seed 调试
CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
  --algo awac --env_name antmaze-umaze-v0 \
  --exp_name awac_antmaze_umaze_seed1 --use_wandb --seed 1

# 3 seed 并排（各占一张卡）
CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
  --algo awac --env_name antmaze-umaze-v0 \
  --exp_name awac_antmaze_umaze_seed1 --use_wandb --seed 1
CUDA_VISIBLE_DEVICES=1 python cs224r/scripts/run_algo.py \
  --algo awac --env_name antmaze-umaze-v0 \
  --exp_name awac_antmaze_umaze_seed2 --use_wandb --seed 2
CUDA_VISIBLE_DEVICES=2 python cs224r/scripts/run_algo.py \
  --algo awac --env_name antmaze-umaze-v0 \
  --exp_name awac_antmaze_umaze_seed3 --use_wandb --seed 3
```

日志写在 `hw3/data/hw3_<exp_name>_<env>_<timestamp>/`。PointMass 轨迹图是该目录下的 `eval_last_traj.png`。

---

## Phase 0: 环境、账号与预习

### 0.1 Mac：只做开发，不训练

本机 **可以**做的：填 TODO、写 LaTeX、编译 PDF、看 WandB。  
本机 **不要**做的：`pip install -r hw3/requirements.txt`、跑 `run_algo.py`。trainer 顶层 `import d4rl`，依赖 Linux + `mujoco_py`。

- [ ] 本机继续用现有环境写代码即可（不必新建 `cs224r-hw3`）
- [ ] 本机 `wandb login`（方便看服务器打上来的曲线；**不要把 key 发到聊天里**）

### 0.2 4090 服务器：一次性训练环境（给服务器 agent 的安装任务）

> **这段就是之后丢给服务器 agent 的说明书。** 目标：在 **其中一张 4090** 上下面这段 smoke test 全绿。6 张卡共用这一个 conda。  
> **不要**复用 `cs224r-hw2-local`（HW2 是新 PyTorch + `gym==0.26` + Meta-World；HW3 要 `gym==0.23.1` + D4RL）。  
> **不要**在 OneDrive / 网络盘上训练；把仓库 rsync 或 git clone 到服务器本地盘。

#### 成功标准（agent 做完必须打印出来）

```text
cuda True NVIDIA GeForce RTX 4090
mujoco_py <path>
d4rl <path>
gym 0.23.1
antmaze-umaze-v0 made
d4rl dataset transitions: <N>   # N 应是几十万量级，不是 0
```

#### 不要做的事

- 不要 `conda activate cs224r-hw2-local` 然后往里面装 D4RL
- 不要直接 `pip install -r requirements.txt`（里面的 `torch==1.13.1` 会从 PyPI 拉到 CPU/错 CUDA 轮子）
- 不要改五个算法文件以外的训练逻辑；安装阶段只动环境
- `mujoco_py` / D4RL 死磕超过 **约 2 小时** 仍失败：停，回报卡点，改走 Modal Starter，不要换 5090 硬装同一套旧栈

#### 步骤 A — 系统依赖和 MuJoCo 2.1.0（HW2 做过就跳过）

```bash
# Ubuntu；包名按发行版微调
sudo apt-get update
sudo apt-get install -y libglew-dev patchelf libosmesa6-dev \
    libgl1-mesa-glx libglfw3 libglew2.2 libegl1-mesa \
    build-essential wget git curl

# 若 ~/.mujoco/mujoco210 已存在（HW2 装过），跳过下载
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p ~/.mujoco && cd ~/.mujoco
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xzf mujoco210-linux-x86_64.tar.gz
fi

# 写入 ~/.bashrc（逐条检查，HW2 可能已经有 MuJoCo 那两行）
grep -q 'MUJOCO_PY_MUJOCO_PATH' ~/.bashrc || \
  echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc
grep -q 'mujoco210/bin' ~/.bashrc || \
  echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
grep -q 'MUJOCO_GL' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
grep -q 'D4RL_SUPPRESS_IMPORT_ERROR' ~/.bashrc || \
  echo 'export D4RL_SUPPRESS_IMPORT_ERROR=1' >> ~/.bashrc
source ~/.bashrc
```

`D4RL_SUPPRESS_IMPORT_ERROR=1` 必须有：D4RL 会 import 一堆可选环境，缺一个就整包炸。

#### 步骤 B — 新 conda，按顺序装（不要 `pip install -r`）

```bash
conda create -n cs224r-hw3 python=3.10.19 -y
conda activate cs224r-hw3

# 1) 先装能在 4090 上跑的 torch。优先课程 pin：1.13.1 + cu117
#    4090 (sm_89) 对 cu117 一般走 sm_86 兼容；驱动是 CUDA 12.x 也没关系，轮子自带 runtime
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 若上一步 import torch 后报 sm_89 / no kernel image：不要死磕 1.13，改成和 HW2 一样的新 torch，算法代码不用改
# pip install torch torchvision torchaudio   # cu121 / cu124 按机器驱动选

# 2) 编译 mujoco_py 之前必须钉死 Cython 0.29
pip install "numpy==1.26" "Cython==0.29.37" "gym==0.23.1" "protobuf==3.20.1"

# 3) 其余课程依赖（先于 D4RL）
pip install "matplotlib==3.5.3" "moviepy==1.0.0" "pyvirtualdisplay==1.3.2" \
    opencv-python-headless "networkx==2.5" "ipdb==0.13.3" scipy \
    "imageio-ffmpeg==0.6.0" "dm_control==1.0.37" "tqdm==4.67.3" \
    "wandb==0.25.0"

# 4) mujoco_py：第一次会本地编译。失败多半是缺 A 的 apt 或没 source bashrc
pip install "mujoco-py==2.1.2.14"

# 5) D4RL 必须用作业 pin 的 commit（不要 pip install d4rl 最新版）
pip install "D4RL @ git+https://github.com/Farama-Foundation/d4rl@89141a689b0353b0dac3da5cba60da4b1b16254d"
```

#### 步骤 C — 代码放到本地盘并 `pip install -e .`

```bash
# 示例：从 Mac rsync（在 Mac 上执行；排除 data/ 和 __pycache__）
# rsync -av --exclude data/ --exclude '__pycache__' --exclude '.git' \
#   "/Users/kaitang/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/Course/Stanford Deep Reinforcement Learning/hw3_offlinerl/" \
#   USER@SERVER:~/cs224r-hw3/hw3_offlinerl/

cd ~/cs224r-hw3/hw3_offlinerl/hw3    # 以实际路径为准
pip install -e .
```

`setup.py` 只声明包名 `cs224r`，必须从 `hw3/` 做 editable install，否则 `from cs224r...` 会挂。

#### 步骤 D — smoke test（agent 必须跑，并把输出贴回来）

```bash
conda activate cs224r-hw3
source ~/.bashrc
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

python - <<'PY'
import gym, torch, mujoco_py, d4rl
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch", torch.__version__)
print("gym", gym.__version__)
print("mujoco_py", mujoco_py.__file__)
print("d4rl", getattr(d4rl, "__file__", d4rl))

env = gym.make("antmaze-umaze-v0")
print("antmaze-umaze-v0 made", env.observation_space, env.action_space)
ds = d4rl.qlearning_dataset(env)
print("d4rl dataset transitions:", len(ds["observations"]))
print("obs dim", ds["observations"].shape, "act dim", ds["actions"].shape)
env.close()
PY
```

第一次跑会把 AntMaze 数据下到 `~/.d4rl/datasets/`，可能要几分钟到十几分钟。`transitions` 为 0 或 `gym.make` 失败都算没过。

- [ ] smoke test 打印出 4090 名 + `gym 0.23.1` + umaze 数据集条数
- [ ] 服务器 `wandb login`（脚本要 `--use_wandb` 才打点；没登录会在 init 时报错）
- [ ] 作业目录已在服务器**本地盘**（不是 OneDrive 挂载）

#### 常见失败怎么处理（给 agent）

| 现象 | 处理 |
| ---- | ---- |
| `mujoco_py` 编译缺头文件 / `GLEW` | 回到步骤 A 补 apt，`source ~/.bashrc` 后再装一次 |
| `import d4rl` 因某个子环境 ImportError | 确认 `D4RL_SUPPRESS_IMPORT_ERROR=1` 已 export |
| `gym.make('antmaze-umaze-v0')` 找不到 | D4RL 没装上，或 gym 版本不是 0.23.x（0.26 / gymnasium 会挂） |
| 数据集下载 HTTP 403 / 超时 | 看 `~/.d4rl/datasets/` 是否不完整；重试 `qlearning_dataset`；URL 挂了再搜该 commit 的 mirror，文件仍放到 `~/.d4rl/datasets/` |
| `torch` 报 sm_89 / no kernel image | 卸 1.13.1，改装 HW2 那套新 torch，**不要**为这改算法 |
| `pip install -e .` 后仍 `No module named cs224r` | 确认 cwd 是 `hw3/`（里面有 `setup.py` 和 `cs224r/`），且当前就是 `cs224r-hw3` |

### 0.3 服务器训练注意（HW3 没有 HW2 那种 demo 路径补丁）

HW3 入口是 `run_algo.py`，不 copy `/root/demos`，**不用改 trainer**。

- [ ] 长任务放 `tmux` / `screen`
- [ ] 用 `CUDA_VISIBLE_DEVICES` 绑单卡；`--which_gpu 0` 保持默认（visible 之后的 0 号）
- [ ] 三个 seed 用三个 `--exp_name ..._seed{1,2,3}`，方便之后对 CSV
- [ ] 必须在 `hw3/` 目录启动（PointMass 的 `--offline_dataset offline_datasets/...` 是相对路径）
- [ ] eval 渲染依赖 `MUJOCO_GL=egl`。若只是视频失败、数字还在刷，可以先忽略视频；PointMass 轨迹图不依赖 MuJoCo 渲染

### 0.4 备选：Modal（默认跳过）

只有 4090 上 D4RL / `mujoco_py` 超过 ~2 小时仍失败时才用。Starter 绑卡后每月 $30 免费额度，按 T4 ≈ $0.59/h 估，整份 HW3 约 20 GPU·小时 ≈ $12，通常不用额外充。

若走这条：本机 `pip install modal` → `modal setup` → `modal secret create wandb WANDB_API_KEY=... --force`，再用 `modal_train.py` / `modal_train_para.py`。自学没有课程兑换链接。

### 0.5 通读材料

- [ ] 通读 [`hw3/README.md`](hw3/README.md)（官方 Modal 入口；自学改走 `run_algo.py`）
- [ ] 通读作业 PDF Overview + Setup + Autograder + 提交格式
- [ ] 通读只读文件（理解 pipeline，**不要改算法逻辑**）:
  - [ ] `cs224r/scripts/run_algo.py` — `--algo {bc,awac,iql}`；AntMaze `num_timesteps=300000`、`scalar_log_freq=50000`；PointMass `50000` / `2000` / 20 eval episodes
  - [ ] `cs224r/infrastructure/rl_trainer_awac.py` — 离线 buffer、eval、WandB；AntMaze 的 return 是 `(reward - 1).sum()`；PointMass 轨迹图 `eval_last_traj.png`
  - [ ] `cs224r/agents/bc_agent.py` — Filtered BC 直接复用，`--filter_top_percent 10` 在 loader 里筛轨迹
  - [ ] `cs224r/infrastructure/utils.py` — PointMass / AntMaze 的 `q_func`、`v_func`（IQL 的 V 只吃 `ob_dim`）
  - [ ] `cs224r/policies/MLP_policy.py` 里已实现的 `MLPPolicy.forward` / `MLPPolicyBC.update`（AWR 只要填权重）
  - [ ] （可选，仅当走 Modal 退路）`modal_train.py`、`modal_train_para.py`、`modal_config.py`
- [ ] 编译一次报告模板确认 LaTeX 可用:
  ```bash
  cd hw3_offlinerl
  latexmk -pdf CS224R_2026_Homework_3.tex
  ```

**关键概念速记**

| 项目 | 值 |
| ---- | -- |
| 任务 | D4RL AntMaze（连续）+ PointMass（离散 gridworld） |
| AntMaze 奖励 | 稀疏：到 goal 为 0，否则 -1（鼓励尽快到） |
| PointMass 奖励 | 同样稀疏：每步 -1，到 goal 为 0 |
| Eval | `Eval_AverageReturn` = eval episode return 的均值；报告取 **最终 checkpoint**，再跨 3 seed 算 mean/std |
| AWAC umaze | `antmaze-umaze-v0`，300k steps，~1h，3 seeds |
| AWAC medium | `antmaze-medium-diverse-v0`，300k steps，~1.5h，3 seeds |
| IQL umaze | 同上，扫 `ζ ∈ {0.2, 0.9}`，各 3 seeds |
| IQL medium | 用更好的 ζ，3 seeds |
| Stitching 数据 | `offline_datasets/pointmass_stitching_dataset.npz`；数据集 max **-46**、avg **-104** |
| PointMass 训练 | `PointmassMedium-v0`，50k steps；IQL ~15min，Filtered BC ~10min |
| Actor 温度 λ | `awac_lambda=0.1`（AWAC 和 IQL 共用 `MLPPolicyAWAC`） |
| Autograder | 只收 P1 两套 + P2 前两套的 CSV；P2.1 **只交最好 ζ**；stitching **不进** autograder |

---

## Phase 1: Problem 1 — Advantage-Weighted Actor-Critic（2 分）

文件:

- [`hw3/cs224r/policies/MLP_policy.py`](hw3/cs224r/policies/MLP_policy.py)
- [`hw3/cs224r/critics/awac_critic.py`](hw3/cs224r/critics/awac_critic.py)
- [`hw3/cs224r/agents/awac_agent.py`](hw3/cs224r/agents/awac_agent.py)

Actor 目标（优势加权 NLL）:

\[
L_\pi(\psi)=-\mathbb{E}_{(s,a)\sim\mathcal{D}}\Big[\log\pi_\psi(a\mid s)\,\exp\big(\tfrac{1}{\lambda}\mathcal{A}^{\pi_k}(s,a)\big)\Big]
\]

Critic 用 clipped double Q；target 动作用 **当前策略** 采样 \(a'\sim\pi(\cdot\mid s')\)（不是数据集里的 next action）:

\[
y = r + \gamma\,(1-d)\,\min_{i\in\{1,2\}} Q_{\bar\phi_i}(s', a')
\]

两个 Q 都对同一个 \(y\) 做 MSE。`update_target_network()` 已经写好（EMA，`τ=0.005`）。

IQL 的 actor 也走 `MLPPolicyAWAC`，所以 **这段 AWR loss 必须先写对**。

### 1.1 实现代码

- [ ] `MLPPolicyAWAC.update`：用 `adv_n` 和 `self.lambda_awac` 算指数权重

\[
w = \exp(A / \lambda)
\]

  `log_prob_n` 已经算好；后面会 `clamp(max=50)` 再做 `-(log_prob * w).mean()`。这里 **只填 `exp_weights`**，不要自己再 clamp / 不要改 loss 公式。

- [ ] `AWACCritic.update`：clipped double-Q TD
  1. `_get_q_value` 算 `Q1(s,a)`、`Q2(s,a)`
  2. `torch.no_grad()` 里 `get_target_q(next_ob_no, next_actions)` 得到 \(\min \bar Q(s',a')\)
  3. terminal 要 mask：`y = r + γ * target * (1 - done)`
  4. `loss` / `loss2` 分别是两个 Q 对 **同一个** `y` 的 MSE（可用 `self.mse_loss`）
  5. 后面已经 `(loss + loss2).backward()`，不要自己 `optimizer.step`

- [ ] `AWACAgent.estimate_advantage`：`A(s,a) = Q(s,a) - V(s)`
  - `Q(s,a)`：`self.critic.get_q(ob_no, ac_na)`（已经是 min 双 Q）
  - `V(s)`：从当前 actor 采样 **一个** \(a\sim\pi(\cdot\mid s)\)，再 `get_q(s, a)`（单样本估计）
  - 整个函数已包在 `torch.no_grad()` 里

- [ ] `AWACAgent.train` 两处:
  1. `next_actions`：对 `next_ob_no` 用当前 actor 采样 \(a'\sim\pi(\cdot\mid s')\)（`self.actor(...)` 返回 distribution，再 `.sample()`）。discrete 是 `[B]`，continuous 是 `[B, ac_dim]`
  2. actor：`estimate_advantage` → `self.actor.update(ob_no, ac_na, adv_n=adv)`；`actor_loss` 就是这个返回值

**自测**（没有 HW2 那种 pytest）: 实现后不应再是 `None`；单 seed 跑起来 critic/actor loss 有限，WandB 的 `Eval_AverageReturn` 在刷且大致上升。

### 1.2 跑实验 A：antmaze-umaze-v0（1 分）

先单 seed 调试，再 3 张 4090 并排。官方 T4 估 ~1h / seed；4090 通常更快。

- [ ] 调试（确认代码没炸）:
  ```bash
  conda activate cs224r-hw3
  cd ~/cs224r-hw3/hw3_offlinerl/hw3
  export MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-umaze-v0 \
    --exp_name awac_antmaze_umaze_seed1 --use_wandb --seed 1
  ```
- [ ] 打开 WandB，确认 `Eval_AverageReturn` 在刷（umaze 每 50k steps 记一次，300k 共约 6 个点）
- [ ] 正式 3 seeds（三个 tmux，各一张卡）:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-umaze-v0 \
    --exp_name awac_antmaze_umaze_seed1 --use_wandb --seed 1
  CUDA_VISIBLE_DEVICES=1 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-umaze-v0 \
    --exp_name awac_antmaze_umaze_seed2 --use_wandb --seed 2
  CUDA_VISIBLE_DEVICES=2 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-umaze-v0 \
    --exp_name awac_antmaze_umaze_seed3 --use_wandb --seed 3
  ```
- [ ] 每个 seed 取 **最终 checkpoint** 的 `Eval_AverageReturn`，再算 3 个数的 mean / std（跨 seed，不是 run 内部）

**记录区 — Problem 1A（umaze）**

| 项目 | 值 |
| ---- | -- |
| WandB run URLs (seed 1/2/3) | |
| seed1 最终 Eval_AverageReturn | |
| seed2 最终 Eval_AverageReturn | |
| seed3 最终 Eval_AverageReturn | |
| mean | |
| std | |
| CSV | `csv_data/P1/1/awac_umaze_seed{1,2,3}.csv` |

- [ ] 从 WandB 图名恰好为 `Eval_AverageReturn` 的图导出 CSV（**一 seed 一文件**，不要合并）

### 1.3 跑实验 B：antmaze-medium-diverse-v0（1 分）

官方 T4 约 **1.5 小时 / seed**。

- [ ] 3 seeds:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-medium-diverse-v0 \
    --exp_name awac_antmaze_medium_diverse_seed1 --use_wandb --seed 1
  CUDA_VISIBLE_DEVICES=1 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-medium-diverse-v0 \
    --exp_name awac_antmaze_medium_diverse_seed2 --use_wandb --seed 2
  CUDA_VISIBLE_DEVICES=2 python cs224r/scripts/run_algo.py \
    --algo awac --env_name antmaze-medium-diverse-v0 \
    --exp_name awac_antmaze_medium_diverse_seed3 --use_wandb --seed 3
  ```
- [ ] 同样取最终 checkpoint，跨 3 seed 算 mean / std

**记录区 — Problem 1B（medium-diverse）**

| 项目 | 值 |
| ---- | -- |
| WandB run URLs (seed 1/2/3) | |
| seed1 / seed2 / seed3 最终 return | |
| mean | |
| std | |
| CSV | `csv_data/P1/2/awac_medium_maze_seed{1,2,3}.csv` |

- [ ] 导出 3 份 CSV

### 1.4 写报告

- [ ] 填 Table `awac_antmaze`（umaze 的 mean / std）
- [ ] 填 Table `awac_antmaze_medium_diverse`

---

## Phase 2: Problem 2 — Implicit Q-Learning（5 分）

文件:

- [`hw3/cs224r/critics/iql_critic.py`](hw3/cs224r/critics/iql_critic.py)
- [`hw3/cs224r/agents/iql_agent.py`](hw3/cs224r/agents/iql_agent.py)

IQL 的 actor 和 AWAC 一样（advantage-weighted NLL，已在 Phase 1 实现）。差别在 critic：

- 单独的 \(V_\phi(s)\)，用 **expectile** 回归 \(\bar Q(s,a)\)
- Q 的 backup 是 \(y = r + \gamma V(s')\)，**不采样** \(a'\sim\pi\)，只用数据集里见过的 \((s,a)\)

\[
L_2^\zeta(\mu)=\lvert\zeta-\mathbbm{1}\{\mu\le 0\}\rvert\,\mu^2
\]

\[
L_V(\phi)=\mathbb{E}_{(s,a)\sim D}\big[L_2^\zeta(\bar Q(s,a)-V_\phi(s))\big]
\]

\[
L_Q(\theta)=\mathbb{E}_{(s,a,s')\sim D}\big[(r+\gamma V_\phi(s')-Q_\theta(s,a))^2\big]
\]

\(\zeta>0.5\) 时更重视 **大于** 当前 V 的 TD target（往上偏），适合从混合质量数据里抠高价值片段。

### 2.1 实现代码

- [ ] `IQLCritic.__init__`：定义 `self.v_net`
  - 用 `v_network_initializer(self.ob_dim)`（**只传 ob_dim**，对照上面 Q_net 的 `.to(ptu.device)`）
  - 下面立刻会 `self.v_optimizer = ... self.v_net.parameters()`，名字必须是 `self.v_net`

- [ ] `expectile_loss(diff)`：返回 **逐样本** `[B]`，不要在这里 `.mean()`

\[
L = \lvert\zeta - \mathbbm{1}\{\delta \le 0\}\rvert \cdot \delta^2,\quad \zeta=\texttt{self.iql\_expectile}
\]

- [ ] `update_v`：`value_loss` = expectile(\(Q_{\bar\theta}(s,a) - V(s)\)) 再 mean  
  `q_t_values` 和 `v_t` 已经算好

- [ ] `update_q`：IQL Bellman，**用 V 不用 Q(s',a')**
  1. `torch.no_grad()`：`v_net(next_ob_no)` → \(V(s')\)
  2. `y = r + γ V(s') (1 - done)`
  3. 两个 Q 都对同一个 `y` 做 MSE → `loss`、`loss2`

- [ ] `IQLAgent.estimate_advantage`：`adv = q_sa - v_pi`（两个量已经算好）

- [ ] `IQLAgent.train` 的 actor 更新：和 AWAC 一样  
  `estimate_advantage` → `self.actor.update(ob_no, ac_na, adv_n=adv)`  
  注意顺序已经写好：先 `update_v`，再 actor，再 `update_q`，再 soft-update target Q

**自测**: `v_net` 存在；单 seed IQL umaze 的 V loss / Q loss 有限，eval return 在动。

### 2.2 跑实验：umaze 扫 ζ = 0.2 / 0.9（2 分）

每个 ζ、每个 seed 官方约 1 小时。6 卡可以把两个 ζ 的 3 seed **一起**挂（6 个 job）。

- [ ] ζ = 0.2（3 seeds）:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.2 \
    --exp_name iql_zeta_0.2_umaze_seed1 --use_wandb --seed 1
  CUDA_VISIBLE_DEVICES=1 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.2 \
    --exp_name iql_zeta_0.2_umaze_seed2 --use_wandb --seed 2
  CUDA_VISIBLE_DEVICES=2 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.2 \
    --exp_name iql_zeta_0.2_umaze_seed3 --use_wandb --seed 3
  ```
- [ ] ζ = 0.9（3 seeds）:
  ```bash
  CUDA_VISIBLE_DEVICES=3 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.9 \
    --exp_name iql_zeta_0.9_umaze_seed1 --use_wandb --seed 1
  CUDA_VISIBLE_DEVICES=4 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.9 \
    --exp_name iql_zeta_0.9_umaze_seed2 --use_wandb --seed 2
  CUDA_VISIBLE_DEVICES=5 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-umaze-v0 --iql_expectile 0.9 \
    --exp_name iql_zeta_0.9_umaze_seed3 --use_wandb --seed 3
  ```
  调试先只跑 `CUDA_VISIBLE_DEVICES=0 ... --seed 1`。
- [ ] 每个 ζ 取 3 个 seed 最终 `Eval_AverageReturn` 的 mean / std
- [ ] 2–3 句：哪个 ζ 更好、为什么  
  提示：ζ=0.2 **低于**中位数，会偏向低价值；ζ=0.9 才是 IQL 想要的「上偏 expectile」，更容易从 suboptimal 数据里估高价值 backup

**记录区 — Problem 2.1（IQL umaze）**

| ζ | seed1 | seed2 | seed3 | mean | std | WandB |
| - | ----- | ----- | ----- | ---- | --- | ----- |
| 0.2 | | | | | | |
| 0.9 | | | | | | |

- 更好的 ζ: ________
- 一句话原因:

- [ ] Autograder 用的 CSV **只导出更好那一组** 的 3 个 seed → `csv_data/P2/1/iql_umaze_seed{1,2,3}.csv`  
  差的那组仍要写进报告表格，但不要放进 `P2/1/`

### 2.3 跑实验：medium-diverse + 对比 AWAC（2 分）

用 2.2 里更好的 ζ。官方 T4 约 1.5h / seed。

- [ ] 3 seeds:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-medium-diverse-v0 \
    --iql_expectile {better_zeta} \
    --exp_name iql_zeta_{better_zeta}_medium_diverse_seed1 --use_wandb --seed 1
  CUDA_VISIBLE_DEVICES=1 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-medium-diverse-v0 \
    --iql_expectile {better_zeta} \
    --exp_name iql_zeta_{better_zeta}_medium_diverse_seed2 --use_wandb --seed 2
  CUDA_VISIBLE_DEVICES=2 python cs224r/scripts/run_algo.py \
    --algo iql --env_name antmaze-medium-diverse-v0 \
    --iql_expectile {better_zeta} \
    --exp_name iql_zeta_{better_zeta}_medium_diverse_seed3 --use_wandb --seed 3
  ```
- [ ] 填 medium 的 mean / std
- [ ] 填对比表：umaze / medium-diverse × AWAC / IQL（都用最终 mean）
- [ ] 3 句：更难的任务（更长 horizon、更大地图）上谁更好，接到 **OOD 动作** 和 **value estimation**  
  提示草稿（写报告前自己改，不要原文照抄）:
  - AWAC 的 Q backup 要 \(a'\sim\pi(\cdot\mid s')\)，策略一偏出数据集，\(Q(s',a')\) 就可能高估
  - IQL 的 Q 只在数据集 \((s,a)\) 上回归，next-state 用 \(V(s')\)，避免对未见过的 a' 查询 Q
  - expectile V 进一步往高价值轨迹偏，medium 这种 stitching / 长程任务通常更吃香

**记录区 — Problem 2.2**

| 项目 | 值 |
| ---- | -- |
| 使用的 ζ | |
| seed1 / 2 / 3 最终 return | |
| mean | |
| std | |
| CSV | `csv_data/P2/2/iql_medium_maze_seed{1,2,3}.csv` |

对比表草稿（填最终 mean）:

| | AWAC | IQL |
| - | ---- | --- |
| antmaze-umaze | | |
| antmaze-medium-diverse | | |

- 差异解释草稿:

### 2.4 PointMass stitching：IQL vs Filtered BC（1 分）

数据集 `offline_datasets/pointmass_stitching_dataset.npz`：max return **-46**，avg **-104**。  
问的是：IQL 能不能把不同轨迹的好片段 **缝** 成一条比数据里任何一条都好的路（return **高于 -46**）。Filtered BC 只模仿 return 最高的 10% 轨迹，**不会** stitch。

- [ ] IQL（用更好的 ζ，3 seeds，约 15min）:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python cs224r/scripts/run_algo.py \
    --algo iql --env_name PointmassMedium-v0 \
    --iql_expectile {better_zeta} \
    --exp_name iql_zeta_{better_zeta}_stitching_seed1 \
    --offline_dataset offline_datasets/pointmass_stitching_dataset.npz \
    --use_wandb --seed 1
  # seed 2 / 3：换 GPU 和 --exp_name / --seed
  ```
- [ ] Filtered BC（3 seeds，约 10min）:
  ```bash
  CUDA_VISIBLE_DEVICES=3 python cs224r/scripts/run_algo.py \
    --algo bc --env_name PointmassMedium-v0 \
    --exp_name filtered_bc_stitching_seed1 \
    --offline_dataset offline_datasets/pointmass_stitching_dataset.npz \
    --filter_top_percent 10 --use_wandb --seed 1
  ```
- [ ] 每个算法、每个 seed：最终 checkpoint 的 `Eval_AverageReturn` 和 `Eval_MaxReturn`；再跨 seed 报 mean ± std
- [ ] 在服务器 `hw3/data/hw3_<exp_name>_.../` 里找 **`eval_last_traj.png`**（`set_logdir(.../eval_)` → `eval_last_traj.png`）。每个算法选 **一个 seed** 的最终图即可。
- [ ] 分析：IQL 是否 > -46；和 Filtered BC 比谁高；是否表现出 stitching

**记录区 — Problem 2.3**

| Algorithm | Average Return ± std | Maximum Return ± std | WandB |
| --------- | -------------------- | -------------------- | ----- |
| IQL | ± | ± | |
| Filtered BC | ± | ± | |

| 图 | 路径 |
| -- | ---- |
| IQL 轨迹 | `hw3/data/.../eval_last_traj.png` |
| Filtered BC 轨迹 | |

- IQL 最终 avg 是否 > -46:
- stitching 一句话:

---

## Phase 3: 收尾

### 3.1 报告 PDF

- [ ] 填写页眉：SUNet ID、Name、Collaborators（自学可写自己的）
- [ ] 检查所有红色 `\answer{}` 已换成数字 / 表 / 图 / 文字
- [ ] P2.1 两个 ζ 都填了；解释写了
- [ ] P2.2 对比表 + 3 句 OOD / value
- [ ] P2.3 两张轨迹图已 `\includegraphics`
- [ ] 编译:
  ```bash
  cd hw3_offlinerl
  latexmk -pdf CS224R_2026_Homework_3.tex
  ```
- [ ] 通读最终 PDF

### 3.2 代码与 CSV 归档（自学备份；官方 zip 格式如下）

```text
submit.zip
├── csv_data/
│   ├── P1/
│   │   ├── 1/
│   │   │   ├── awac_umaze_seed1.csv
│   │   │   ├── awac_umaze_seed2.csv
│   │   │   └── awac_umaze_seed3.csv
│   │   └── 2/
│   │       ├── awac_medium_maze_seed1.csv
│   │       ├── awac_medium_maze_seed2.csv
│   │       └── awac_medium_maze_seed3.csv
│   └── P2/
│       ├── 1/          # 仅最好 ζ，不要两个 ζ 混放
│       │   ├── iql_umaze_seed1.csv
│       │   ├── iql_umaze_seed2.csv
│       │   └── iql_umaze_seed3.csv
│       └── 2/
│           ├── iql_medium_maze_seed1.csv
│           ├── iql_medium_maze_seed2.csv
│           └── iql_medium_maze_seed3.csv
└── cs224r/             # 与仓库同结构的 .py（已填 TODO）
```

CSV 规则（autograder 会卡）:

- 必须从 WandB **名为 `Eval_AverageReturn` 的那张图**导出，不要下别的 metric
- **一 seed 一文件**，每个子目录恰好 3 个 CSV，且是同一超参
- stitching / 较差的那个 ζ **不进** `csv_data/`

- [ ] 整理 `hw3_submission/`（或同等目录）
- [ ] 确认 12 份 CSV 来自**最终**那几轮 run，不是调试 seed

---

## Phase 4: 完成后 — 多模型 Review 打分

> 你做完后回来找我，我会开**多个 agent、使用不同模型**分别 review，再汇总打分。

### 你需要提供

1. 代码：`MLP_policy.py`，`awac_critic.py`，`awac_agent.py`，`iql_critic.py`，`iql_agent.py`
2. 12 份 autograder CSV +（可选）stitching 的 WandB / `eval_last_traj.png`
3. 编译好的 PDF 报告
4. （可选）WandB run 链接 / 服务器 `hw3/data/` 日志路径

### Review 维度（按官方 7 分）

| 维度 | 分值 | 检查内容 |
| ---- | ---- | -------- |
| P1 AWAC 代码 | （含在实验里） | AWR 权重、clipped double-Q、terminal mask、单样本 V、\(a'\sim\pi\) |
| P1A umaze | 1 | 3 seed 最终 mean ± std；CSV 齐 |
| P1B medium | 1 | 同上 |
| P2 IQL 代码 | （含在实验里） | `v_net`、expectile、\(L_V\)、\(y=r+\gamma V(s')\)、A=Q-V |
| P2.1 ζ 扫描 | 2 | 0.2 / 0.9 都有数；解释哪个更好；autograder 只交最好 ζ |
| P2.2 medium + 对比 | 2 | IQL medium 数字；AWAC vs IQL 表；3 句接到 OOD / value |
| P2.3 stitching | 1 | avg/max ± std；两张轨迹图；是否 > -46 以及为何 IQL 更能 stitch |
| **合计** | **7** | |

### 触发方式

完成后直接说：

> 「HW3 做完了，帮我打分 review」

并附上代码路径、PDF、CSV/截图位置即可。

---

## 推荐实现顺序

```
服务器 agent：Phase 0.2 装 cs224r-hw3 + smoke test
    ↓
Mac: exp_weights → AWAC TD → AWAC advantage → AWAC train
    ↓  4090 GPU0 单 seed 调试 umaze
Mac: v_net → expectile → update_v → update_q → IQL advantage / actor
    ↓  4090 GPU0 单 seed 调试 IQL umaze
4090: GPU0–2 AWAC umaze 3 seed
      可同时 GPU3–5 挂 IQL ζ=0.2 或等 AWAC 结束再扫两个 ζ
    ↓
4090: AWAC medium 3 seed + IQL medium 3 seed（更好的 ζ）
    ↓
4090: PointMass IQL + Filtered BC
写报告 → 整理 12 份 CSV → 编译 PDF → 找我 review
```

**并行建议**: 先单 seed 证明 AWAC / IQL 都能跑，再开 3 张卡。不要在代码还 `None` 的时候占满 6 卡。medium 比 umaze 长，等 ζ 选定再跑 IQL medium。6 卡不要用来拆一个 job，只用来并排独立 seed。

无 HW2 那种本机单测。自测标准：不再是 `None` / `pass`；critic、actor loss 有限；`Eval_AverageReturn` 随训练上升。

---

## 进度总览

| Phase | 内容 | 状态 |
| ----- | ---- | ---- |
| 0 | 4090 环境 + WandB + 预习 | ⬜ |
| 1 | AWAC 代码 + umaze + medium | ⬜ |
| 2 | IQL 代码 + ζ 扫描 + medium 对比 + stitching | ⬜ |
| 3 | 报告 + CSV 归档 | ⬜ |
| 4 | 多模型 Review | ⬜ |

---

*最后更新: 2026-09-11*
