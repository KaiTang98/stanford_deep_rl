# HW2 两种方法伪代码

本文是学习用的控制流程概要，帮助对应 `train_*.py`、agent 和 replay buffer 的调用关系。它不替代作业要求中的公式推导，也不直接作为三份算法文件的实现答案。

## 1. PPO（on-policy）

对应文件：`hw2 4/on_policy.py`、`hw2 4/train_on_policy.py`

```text
初始化环境、Actor、Critic、优化器
读取 expert demonstrations

用 demonstrations 做若干步行为克隆预训练

while 尚未达到训练帧数:
    如果当前 episode 已结束:
        reset 环境
        保存新的初始 observation

    如果到了评估时间:
        用当前 actor 在若干个 episode 中运行
        记录 success rate

    根据当前 observation:
        actor 产生动作分布
        从分布中采样 action
        保存当时的 action log-probability

    将 action 送入环境：
        next_observation, reward, done, info = env.step(action)

    把这一时刻的数据放入 rollout buffer：
        observation
        action
        reward
        discount / done
        next_observation
        旧策略的 log-probability

    如果 rollout buffer 已达到指定长度:
        用 critic 计算每个 observation 和 next_observation 的 value
        从 rollout 末尾向前计算 GAE advantage
        根据 value 和 advantage 得到训练目标
        对 advantage 做标准化

        重复若干个 PPO epoch:
            打乱 rollout 中的样本

            对每个 minibatch:
                用当前 actor 重新计算 action log-probability
                计算新旧策略的概率比
                使用 PPO 的 ratio clipping 形成 actor loss
                用 critic 计算 value loss
                计算 entropy 与 reference-policy regularization

                将这些项合并并累计梯度

            每个 epoch 结束后更新 actor 和 critic 参数

        清空 rollout buffer

    更新 observation = next_observation
    更新 episode 状态和已用训练帧数
```

要点：PPO 的更新只使用刚刚收集的一段 on-policy rollout；更新完成后，这批数据就不再继续复用。`3A/3B` 不属于 PPO，而是下面的 off-policy 方法。

## 2. Off-policy actor-critic（3A / 3B）

对应文件：`hw2 4/off_policy.py`、`hw2 4/train_off_policy.py`

```text
初始化环境、Actor
初始化 N 个 online critics
为每个 online critic 建立一个 target critic 副本
初始化 actor / critic 优化器
读取 expert demonstrations，并放入 replay buffer

先做指定次数的行为克隆预训练

while 尚未达到训练帧数:
    如果当前 episode 已结束:
        reset 环境
        保存新的初始 observation

    如果到了评估时间:
        用当前 actor 在若干个 episode 中运行
        记录 success rate

    根据 warmup 状态选择 action：
        warmup 阶段使用探索动作
        否则由 actor 根据 observation 采样动作

    重复 UTD 次数：
        从 replay buffer 采样 n-step batch

        对 batch 中的 next_observation：
            用 actor 产生 next_action
            从若干个 target critics 中随机选两个
            取这两个 target critic 估计中较保守的一个
            结合 reward、discount 和终止状态构造 target value

        对每个 online critic:
            用同一个 target value 计算 critic loss
            更新该 critic 的参数

        对每个 target critic:
            用软更新方式向对应的 online critic 靠近

    如果已经过了 warmup 阶段:
        从 replay buffer 采样 actor batch
        actor 根据 observation 产生 action
        用全部 online critics 评估这些 action
        汇总 critic 评价，更新 actor，使其倾向于更高价值的动作
        按配置执行可选的行为克隆正则或 BC 更新

    将当前 transition 写入 replay buffer：
        observation
        action
        reward
        next_observation
        discount / done

    更新 observation、episode 状态和已用训练帧数
```

两组配置的主要区别：

```text
3A: N = 2 个 critics，UTD = 1
3B: N = 10 个 critics，UTD = 5
```

两者共享同一套基本流程：replay buffer → target critics 计算训练目标 → online critics 更新 → actor 根据 critics 更新。区别主要在 critic ensemble 的规模和每个环境步对应的 critic 更新次数。

## 3. 与环境和数据流的对应关系

```text
mw.py reset / step
        │
        ├── observation: 39 维
        │       ├── 当前末端执行器、夹爪、锤子、钉子和目标位置
        │       └── 后 18 维包含上一时刻相关观测
        │
        ├── agent.act(observation)
        │       └── action: 4 维
        │           ├── 前 3 维：末端执行器位移控制
        │           └── 第 4 维：夹爪控制
        │
        └── transition 写入 rollout buffer 或 replay buffer
```

PPO 主要读取一段连续的新 rollout；off-policy 方法主要从 replay buffer 中反复采样历史 transition。因此，前者强调数据必须来自当前策略，后者允许数据被多次复用。

## 4. 学习时的代码阅读入口

建议按以下顺序对照伪代码：

1. `train_on_policy.py`：先看 PPO 的外层训练循环和调用时机。
2. `on_policy.py`：再看 rollout 数据如何进入 GAE、actor 更新和 critic 更新。
3. `train_off_policy.py`：看 warmup、replay buffer 和 UTD 循环。
4. `off_policy.py`：最后逐个对应 BC、critic ensemble、target update 和 actor update。
5. `mw.py`、`replay_buffer.py`：核对 observation、action、done/discount 在数据结构中的实际形状。
