# CS224R HW1 自学进度清单

> **课程**: CS224R Deep Reinforcement Learning — Imitation Learning  
> **作业截止（官方）**: 2026/4/10 9pm PT  
> **你的状态**: 自学，不提交 Gradescope；完成后请 AI 多模型 review 打分

---

## 使用说明

- 按顺序从上到下做，每完成一项把 `[ ]` 改成 `[x]`
- 实验结果填在对应「记录区」，最后汇总进 `[CS224R_2026_Homework_1.tex](CS224R_2026_Homework_1.tex)`
- 代码目录: `[hw1_starter_code/](hw1_starter_code/)`
- 每完成一个大 Problem，建议先跑通对应命令再继续

### 常用命令

```bash
# 激活环境（新终端先执行一次）
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cs224r

# 进入代码目录
cd "Lec1&2_imitationlearning/hw1_starter_code"

# 渲染报错时
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

---



## Phase 0: 环境与预习

- [x] 安装 Miniforge + `cs224r` 环境（Python 3.10）
- [x] 安装依赖：torch, gymnasium, pygame, matplotlib, imageio[ffmpeg], numpy==2.2.4
- [x] 验证 MPS 可用（Apple Silicon GPU）
- [x] 通读 `[README.md](hw1_starter_code/README.md)`（任务、action chunking、环境设定）
- [x] 通读只读文件（理解 pipeline，不要改）：
  - [x] `main.py` — 训练入口
  - [x] `expert.py` — 专家策略与 demo 收集
  - [x] `flappy_bird_env.py` — 环境物理
  - [x] `visualization.py` — 评估与录视频
- [x] 编译一次报告模板确认 LaTeX 可用：
  ```bash
  cd Lec1&2_imitationlearning
  latexmk -pdf CS224R_2026_Homework_1.tex
  ```

**关键概念速记**


| 项目            | 值                                                |
| ------------- | ------------------------------------------------ |
| 观测            | `[dist_to_pipe, gap1_y, gap2_y, bird_y]`，4 维，归一化 |
| 动作            | 目标 y ∈ [0, 1]                                    |
| ACTION_CHUNK  | 20（一次预测 20 步）                                    |
| EXECUTE_STEPS | 10（只执行前 10 步再 re-query）                          |
| 成功            | 存活 1000 步                                        |


---



## Phase 1: Problem 1 — BC + MSE 回归（2 分）



### 1.1 实现代码

文件: `[networks.py](hw1_starter_code/networks.py)`, `[losses.py](hw1_starter_code/losses.py)`

- [x] `BCPolicy.__init__` — 3 层 MLP：Linear → ReLU → Linear → ReLU → Linear → Sigmoid
  - 输入: `state_dim=4`
  - 输出: `action_dim=20`（action chunk）
- [x] `BCPolicy.forward`
- [x] `mse_loss`（作业 PDF 称 `bc_loss`，代码里函数名是 `mse_loss`）
  - \mathcal{L} = \frac{1}{N}\sum_i \hat{a}_i - a_i^*^2

**自测**: 实现后应不再报 `NotImplementedError`

### 1.2 跑实验

- [x] Easy mode:
  ```bash
  python main.py --method bc_reg --env easy
  ```
- [x] Hard mode:
  ```bash
  python main.py --method bc_reg --env hard
  ```

**记录区 — Problem 1 结果**


| 环境   | Mean Episode Length | Std   | 结果文件路径                                |
| ---- | ------------------- | ----- | ------------------------------------- |
| easy | 959.2               | 120.8 | `results/<timestamp>/bc_reg_easy.txt` |
| hard | 284.8               | 56.9  | `results/<timestamp>/bc_reg_hard.txt` |


- [x] 保存/复制 `bc_reg_easy.txt`、`bc_reg_hard.txt` 到方便提交的目录



### 1.3 写报告（`[CS224R_2026_Homework_1.tex](CS224R_2026_Homework_1.tex)`）

- [x] 填 easy mode 结果表（50 eval episodes 的 mean ± std）
- [x] 填 hard mode 结果表
- [x] 写 2–3 句解释：hard 上 MSE 为什么差？
  - 提示：hard 专家靠近管道时**随机**选 gap1/gap2 → 双峰分布；MSE 回归会**平均**两个目标

---



## Phase 2: Problem 2 — Flow Matching（2 分）



### 2.1 实现代码

文件: `[networks.py](hw1_starter_code/networks.py)`, `[losses.py](hw1_starter_code/losses.py)`

- [x] `FlowMatchingSchedule.interpolate`
  - 给定 clean action a_1、timestep \tau，采样 noise \epsilon \sim \mathcal{N}(0,I)
  - a_\tau = \tau a_1 + (1-\tau)\epsilon，目标速度 v = a_1 - \epsilon
- [x] `FlowMatchingSchedule.sample`
  - 从 a_0 \sim \mathcal{N}(0,I) 出发，Euler 积分 `num_steps` 步
  - 结果 clamp 到 [0, 1]
- [x] `flow_matching_loss`
  - 调 `schedule.interpolate`，对预测速度与目标速度做 MSE



### 2.2 跑实验

- [x] Hard mode:
  ```bash
  python main.py --method bc_flow --env hard
  ```

**记录区 — Problem 2 结果**


| 环境   | Mean Episode Length | Std | 结果文件路径                                 |
| ---- | ------------------- | --- | -------------------------------------- |
| hard | 1000                | 0   | `results/<timestamp>/bc_flow_hard.txt` |


- [x] 保存/复制 `bc_flow_hard.txt`



### 2.3 写报告

- [x] 填 hard mode 结果表
- [x] 写 2–3 句解释：flow matching 为什么在 hard 上更好？
  - 提示：生成式模型能建模**多峰**动作分布，不像 MSE 只能输出均值

---



## Phase 3: Problem 3 — DAgger（2 分）



### 3.1 实现代码

文件: `[dagger.py](hw1_starter_code/dagger.py)`

- [x] `DeterministicExpert.act` — hard mode 靠近管道时**确定性地**选 gap1（upper gap）
  - 消除随机双峰，让 MSE 回归能学
- [x] `rollout_episode` — 单 episode rollout
  - `env.reset(seed=seed)`
  - action chunk + receding horizon（`EXECUTE_STEPS=10`）
  - 返回 `(ep_states, ep_expert_actions)` — 注意这里收集的是 policy 访问的 states
- [x] `rollout_and_relabel` — 多 episode
  - 调 `rollout_episode` 收集 states
  - 用 `DeterministicExpert` relabel 成 expert actions
  - window 成 `(state, action_chunk)` 训练对



### 3.2 跑实验

- [x] Hard mode（默认 5 rounds）:
  ```bash
  python main.py --method dagger --env hard
  ```
- [ ] 生成对比图:
  ```bash
  python main.py --plot
  ```

**记录区 — Problem 3 结果**


| Round          | Mean Episode Length | Std |
| -------------- | ------------------- | --- |
| 0 (initial BC) |                     |     |
| 1              |                     |     |
| 2              |                     |     |
| 3              |                     |     |
| 4              |                     |     |
| 5 (final)      |                     |     |



| 方法 (hard)      | Mean | Std |
| -------------- | ---- | --- |
| BC Reg         |      |     |
| Flow Matching  |      |     |
| DAgger (final) |      |     |


- [ ] 保存/复制 `dagger_hard.txt`
- [ ] 保存 DAgger learning curve 图（`plots/` 或自己画）
- [ ] 保存三方法对比 bar chart / table



### 3.3 写报告

- [ ] 插入 DAgger learning curve（x=round, y=mean episode length, error bar=std）
- [ ] 画 BC Reg 水平基准线
- [ ] 插入三方法对比图/表
- [ ] 写 3–4 句解释：
  - DAgger 为什么逐轮变好？
  - 确定性专家的作用？
  - 如何解决 MSE 在 hard 上的挑战？

---



## Phase 4: 收尾



### 4.1 报告 PDF

- [ ] 填写页眉：Name（自学可写自己的）、Collaborators
- [ ] 检查所有红色占位处已替换
- [ ] 编译 PDF:
  ```bash
  cd Lec1&2_imitationlearning
  latexmk -pdf CS224R_2026_Homework_1.tex
  ```
- [ ] 通读最终 PDF，确认表格/图/公式无误



### 4.2 代码整理（自学归档用）

按官方提交格式整理一份备份（即使不上传 Gradescope）：

```text
hw1_submission/
├── hw1/
│   ├── networks.py      # 已填 TODO
│   ├── losses.py        # 已填 TODO
│   ├── expert.py        # 原样（read-only）
│   ├── dagger.py        # 已填 TODO
│   └── ...              # 其他 starter 文件
├── bc_reg_easy.txt
├── bc_reg_hard.txt
├── bc_flow_hard.txt
└── dagger_hard.txt
```

- [ ] 整理 submission 文件夹
- [ ] （可选）跑完整 pipeline 确认一切正常:
  ```bash
  python main.py
  ```

---



## Phase 5: 完成后 — 多模型 Review 打分

> 你做完后回来找我，我会开**多个 agent、使用不同模型**分别 review，再汇总打分。



### 你需要提供

1. 完成的代码：`networks.py`, `losses.py`, `dagger.py`
2. 四个结果文件：`bc_reg_easy.txt`, `bc_reg_hard.txt`, `bc_flow_hard.txt`, `dagger_hard.txt`
3. 编译好的 PDF 报告
4. （可选）`plots/` 里的对比图、DAgger 曲线



### Review 维度（预计）


| 维度              | 分值    | 检查内容                                |
| --------------- | ----- | ----------------------------------- |
| Problem 1 代码    | 0.5   | BCPolicy 结构、MSE loss 正确性            |
| Problem 1 实验+解释 | 1.5   | easy/hard 结果合理、hard 失败原因正确          |
| Problem 2 代码    | 0.5   | interpolate / sample / loss 正确      |
| Problem 2 实验+解释 | 1.5   | hard 结果、flow matching 优势解释          |
| Problem 3 代码    | 0.5   | DeterministicExpert、rollout、relabel |
| Problem 3 实验+解释 | 1.5   | learning curve、对比、DAgger 解释         |
| **合计**          | **6** |                                     |




### 触发方式

完成后直接说：

> 「HW1 做完了，帮我打分 review」

并附上代码路径或改动说明即可。

---



## 推荐实现顺序（官方建议）

```
BCPolicy → mse_loss → bc_reg easy/hard
    ↓
FlowMatchingSchedule.interpolate → .sample → flow_matching_loss → bc_flow hard
    ↓
DeterministicExpert.act → rollout_episode → rollout_and_relabel → dagger hard
    ↓
--plot → 写报告 → 编译 PDF → 找我 review
```

---



## 进度总览


| Phase | 内容            | 状态          |
| ----- | ------------- | ----------- |
| 0     | 环境 + 预习       | 环境 ✅ / 预习 ⬜ |
| 1     | BC + MSE      | ⬜           |
| 2     | Flow Matching | ⬜           |
| 3     | DAgger        | ⬜           |
| 4     | 报告 + 归档       | ⬜           |
| 5     | 多模型 Review    | ⬜           |


---

*最后更新: 2026-09-01*