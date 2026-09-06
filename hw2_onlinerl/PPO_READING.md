# Problem 2 阅读清单：熟悉 `PPOAgent`

只改 `[hw2 4/on_policy.py](hw2%204/on_policy.py)`。其它文件只读，弄清数据从哪来、空该填什么。  
每读完一块，能回答该块的问题再往下；卡住把文件名和行号发出来讨论。

你后面要写的只有三处，都在 `on_policy.py`：

1. `compute_gae` 的反传循环
2. `update` 里用 critic 算 value，再调用 `compute_gae`
3. `update` 里的 `ratio` 和 `policy_loss`（PPO-Clip）

---

## 0. 整张图

这条作业里的 PPO 不是“严格 on-policy”（同一批数据会更新多个 epoch），但数据一定是**刚从当前策略采的**。一轮是：

```
演示 BC 预训练
  → 冻一份 reference actor
  → 采满 4096 步 rollout（记下 old_log_prob）
  → 算 GAE / returns
  → 打乱后多个 epoch 做 PPO-Clip + value + entropy + reverse-KL
  → 丢掉这批数据，再采下一批
```

- [x] 能用自己的话复述上面这一轮
- [x] 知道三处 `YOUR CODE HERE` 分别对应什么

---



## 1. 超参：`cfgs/on_policy_config.yaml`

看 `agent:` 下面这些，它们会进 `PPOAgent` 的 `self.xxx`。


| 名字                                                | 默认                | 读时代入的位置                 |
| ------------------------------------------------- | ----------------- | ----------------------- |
| `clip_eps`                                        | 0.1               | clip 区间 1\pm\varepsilon |
| `ppo_epochs`                                      | 3                 | 同一批 rollout 更新几轮        |
| `gae_lambda` / `gamma`                            | 0.99 / 0.99       | GAE                     |
| `value_coef` / `entropy_coef` / `reverse_kl_coef` | 0.5 / 0.01 / 0.01 | 总 loss 的权重（已写好）         |
| `batch_size`                                      | 64                | minibatch               |
| `rollout_length`                                  | 4096              | 采多少步才 `update` 一次       |
| `pretrain_steps`                                  | 10000             | BC 步数                   |
| `num_train_frames`                                | 1e6               | 服务器上要跑到的长度              |


- [x] 读完 yaml
- [x] `self.clip_eps` 用在哪？
- [x] `self.gae_lambda`、`self.gamma` 用在哪？

---



## 2. 网络：`Actor` → `TruncatedNormal` → `Critic`



### `Actor`（`on_policy.py` 约 13–39 行）

- `forward` 输出的不是一个动作，是 `utils.TruncatedNormal` **分布**（动作截在 [-1,1]）
- `mu = tanh(...)`，`std` 从 `log_std` 来
- 之后会用：`dist.sample()`、`dist.log_prob(action)`、`dist.entropy()`、`dist.mean`

- [x] 读完 `Actor`
- [x] 知道 `forward` 返回的是分布，不是动作向量



### `TruncatedNormal`（`utils.py` 约 111–135 行）

- 只看 `sample`；它继承自 `Normal`（所以有 `log_prob`）
- 不必抠 `_clamp` 的梯度技巧

- [x] 读完 `TruncatedNormal.sample`



### `Critic`（`on_policy.py` 约 42–58 行）

- 这是 **V(s)**，只吃 observation，**不吃 action**
- `forward` 输出 `[batch, 1]`

- [x] 读完 `Critic`
- [x] PPO 的 critic 和 Q-learning 的 Q 表差在哪？
- [x] 为什么 GAE 需要 V(s) 和 V(s')？

---



## 3. `PPOAgent` 已写好的骨架（约 61–111 行）

按方法读，先别跳进 `compute_gae` / `update`。

- [x] `__init__`：一个 Adam 同时更新 actor + critic；`reference_actor = None`
- [x] `train`：切 train/eval
- [x] `set_reference_policy`：深拷贝当前 actor 并冻住（BC 结束立刻调；后面 reverse-KL 用它）
- [x] `act`：eval 用 `dist.mean`，训练采样用 `dist.sample`
- [x] 知道真正采数据时 `train_on_policy.py` 会**直接调** `self.agent.actor`，不完全走 `act`
- [x] `reference_actor` 是哪一时刻的策略？为什么要冻？

---



## 4. 已实现的 `bc`（约 297–322 行）

当作 `log_prob` 范本：

```text
dist = self.actor(obs)
loss = -dist.log_prob(action).sum(-1).mean()
```

连续动作是多维的，所以 `.sum(-1)` **把每个动作维的 log 概率加起来**。后面 PPO 的 `new_log_prob` 已经按同样方式写好了。

- [x] 读完 `bc`
- [x] 为什么是 -\log\pi(a|s)？
- [x] `.sum(-1)` 在对什么求和？

---



## 5. 训练循环：`train_on_policy.py` 的 `Workspace.train`（约 190–269 行）

这是理解「update 吃什么」最重要的一段。

- [x] 先 `agent.bc` 循环 `pretrain_steps` 次
- [x] 然后 `agent.set_reference_policy()`
- [x] 每步用**当前** actor 采样，并记下 `old_log_prob = log π_old(a|s)`
- [x] `discount = env_discount * cfg.discount`
- [x] `rollout_buffer.add(...)`
- [x] buffer 满了 → `agent.update(buffer.get())` → `buffer.reset()`
- [x] `old_log_prob` 是在更新前还是更新后记的？
- [x] 一次 `update` 用的是几步数据？

---



## 6. `RolloutBuffer`（`train_on_policy.py` 约 32–82 行）

看 `add` 存了哪 7 样、`get()` 返回什么。`update` 的 docstring 和这里一一对应：

`obs, action, reward, discount, next_obs, done, old_log_prob`

- [x] `discount` **已经乘过 γ**，不是裸的 0/1
- [x] `done` 是 0/1
- [x] `get()` 包成**长度为 1 的 list**，所以 `update` 里 `for batch in rollout_buffer` 只会转一圈，再 `torch.cat` 得到 `[T, ...]`
- [x] `compute_gae` 的五个参数分别从 buffer 的哪几项、再经过什么计算得到？

---



## 7. 第一处要写：`compute_gae`（约 113–140 行）+ 单测

作业公式（从 t=T-1 倒着走）：

\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)

\hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1},\quad \hat R_t = \hat A_t + V(s_t)

对齐代码：

- 外面已经准备好 `advantages`、`gae`，循环是 `for t in reversed(range(T))`
- 函数**另外**给了 `discounts`。训练时它已是 \gamma \times env_discount；**单测里** `discounts` **恒为 0.99，**`dones` **中间有一个 1**。所以必须用 `dones` 做 (1-d_t)，不能只信 `discounts`
- 循环结束后 `returns = advantages + values` **已经写好**，只填循环体
- 对照 `tests/test_on_policy.py` 的 `test_compute_gae_matches_manual_recursion`

- [x] 读完 `compute_gae` 的签名、参数、已有骨架
- [x] 读完 GAE 单测的输入 / 期望输出
- [x] `done=1` 时 \hat A_{t+1} 还要不要传回去？
- [x] `gae` 这个变量在循环里扮演什么？
- [x] （实现后）`python tests/test_on_policy.py` 的 GAE 测试通过

---



## 8. 第二处要写：`update` 里 GAE 那一块（约 191–205 行）

这段已经包在 `torch.no_grad()` 里（target 不能反传）。需要：

- 用 `self.critic` 从 `obs_all`、`next_obs_all` 得到 `values`、`next_values`
- 调 `self.compute_gae(...)`，得到 `advantages_all` **和** `returns_all`
- 名字必须是这两个：后面标准化、minibatch、value loss 都在用

后面已写好：把 advantage **标准化**（减均值除标准差）。不要改那几行。

- [ ] 读完 `update` 里拼接 `*_all` 张量的部分
- [ ] 知道这里为什么要 `no_grad`
- [ ] `returns_all` 后面给谁当监督信号？
- [ ] （实现后）变量名是 `advantages_all` / `returns_all`

---



## 9. 第三处要写：PPO-Clip（约 233–253 行）

已经算好、直接用：

- `new_log_prob`：当前策略 \log\pi_\theta(a|s)
- `olp_ep`：采集时的 \log\pi_{\text{old}}(a|s)
- `adv_ep`：标准化后的 \hat A
- `self.clip_eps`

要写，且名字固定：

1. `ratio` \rho_t = \exp(\text{new_log_prob} - \text{olp_ep})（用 log 差，不要除概率）
2. `policy_loss`：PPO-Clip，**最小化**所以带负号

L^{\mathrm{CLIP}} = -\mathrm{mean}\Big(\min\big(\rho\hat A,\ \mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A\big)\Big)

下面已经写好、会用到 `ratio`：reverse-KL、`value_loss`、总 `loss`、backward。名字写错会 `NameError`。

可选：把同一段 clip 代码贴进 `test_clipped_surrogate_objective`（把 `self.clip_eps` 改成 `clip_eps`）。

- [ ] 读完 minibatch 循环里已写好的 `new_log_prob` / `entropy`
- [ ] 知道 `ratio` 为什么必须用 \exp(\log\text{差})
- [ ] advantage 为负时 clip 在防什么？
- [ ] （实现后）`ratio`、`policy_loss` 名称正确，后面 reverse-KL 能跑
- [ ] （可选）clip 单测通过

---



## 10. 总 loss（约 259–275 行，只读不改）

```text
loss = policy_loss + c1 * value_loss - c2 * entropy + c3 * reverse_kl
```

多个 minibatch **先累梯度**，一个 epoch 只 `step` 一次。

- [ ] 扫过总 loss 和 `clip_grad_norm_` / `opt.step`
- [ ] 知道一个 epoch 只 step 一次

---

读完并实现 GAE + clip、单测通过之后，再 push 到 4090 训练。