# Expert demo debug: original npz vs open-loop replay

- demo: `/home/ktang/ws/stanford_deep_rl/hw2_onlinerl/hw2 4/demos/demo_4_33.npz`
- replay seed: `0`

## Original npz

- T=33, demo_success=True
- reward_sum=1.0, n_pos=1, last=1.0
- |a|>1 frac=0.515, action range=[-1.598, 2.266]
- nail0=[0.23999999463558197, 0.6399999856948853, 0.10999999940395355] → nail_last=[0.23999999463558197, 0.7241042852401733, 0.10999999940395355] (goal=[0.23999999463558197, 0.7400000095367432, 0.10999999940395355])
- dist nail→goal: 0.1000 → 0.0159
- dist hammer→goal last=0.1931, hand→hammer last=0.0494

## Open-loop replay (same pipeline as dump_expert_progress mp4)

- replay_T=34, **replay_success=False**
- replay_last_reward=0.0, reward_sum=0.0
- replay nail_last=[0.23999999463558197, 0.6399999856948853, 0.10999999940395355], dist nail→goal last=0.1000

## Obs mismatch (original vs replay, aligned length)

- aligned_T=33
- RMSE hand=0.0509, hammer=0.0608, nail=0.0115, full=0.0447

## Interpretation

- `index.md` / plot use **original** demo_success.
- Expert **mp4** uses open-loop replay qpos; its visual outcome is **replay_success**, which can differ.
- Large RMSE means the video is not the recorded expert states.
