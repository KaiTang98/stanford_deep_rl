#!/usr/bin/env python3
"""Read-only debug: original expert npz vs open-loop action replay.

Does NOT modify homework algorithm files or train_*.py / mw.py.
Outputs under viz/retrain_process/expert_debug/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "viz" / "retrain_process" / "expert_debug"
DEFAULT_DEMO = ROOT / "demos" / "demo_4_33.npz"
SAVED_QPOS = ROOT / "viz" / "retrain_process" / "expert" / "demo_4_33_qpos.npz"


def summarize_original(demo_path: Path) -> dict:
    d = np.load(demo_path)
    obs = d["observation"]
    action = d["action"]
    reward = d["reward"].reshape(-1)
    hand = obs[:, 0:3]
    hammer = obs[:, 4:7]
    nail = obs[:, 11:14]
    goal = obs[:, 36:39]
    gripper = obs[:, 3]
    return {
        "path": str(demo_path),
        "T": int(len(reward)),
        "demo_success": bool(reward[-1] > 0),
        "reward_sum": float(reward.sum()),
        "n_pos_reward": int((reward > 0).sum()),
        "reward_last": float(reward[-1]),
        "action_abs_gt1_frac": float((np.abs(action) > 1).any(axis=1).mean()),
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "hand0": hand[0].tolist(),
        "hand_last": hand[-1].tolist(),
        "hammer0": hammer[0].tolist(),
        "hammer_last": hammer[-1].tolist(),
        "nail0": nail[0].tolist(),
        "nail_last": nail[-1].tolist(),
        "goal": goal[-1].tolist(),
        "gripper0": float(gripper[0]),
        "gripper_last": float(gripper[-1]),
        "dist_hand_hammer_last": float(np.linalg.norm(hand[-1] - hammer[-1])),
        "dist_hammer_goal_last": float(np.linalg.norm(hammer[-1] - goal[-1])),
        "dist_nail_goal_last": float(np.linalg.norm(nail[-1] - goal[-1])),
        "dist_nail_goal_0": float(np.linalg.norm(nail[0] - goal[0])),
    }


def open_loop_replay(demo_path: Path, seed: int) -> dict:
    """Random reset + open-loop actions — same idea as dump_expert_progress.replay_demo_qpos."""
    import mw
    from collect_qpos import get_sim
    from visualize_trajectories import load_demo, parse_obs

    demo = load_demo(demo_path)
    np.random.seed(seed)
    env = mw.make()
    # Walk past dm_env wrappers for info if needed
    ts = env.reset()
    sim = get_sim(env)
    obs_list = [ts.observation.copy()]
    rewards = [float(ts.reward)]
    qpos = [sim.data.qpos.copy()]
    last_info_success = None

    # Mirror dump_expert_progress: iterate all stored actions including dummy first row.
    for a in demo["action"]:
        # Prefer unwrapped step to read Meta-World info['success']
        inner = env
        while hasattr(inner, "_env") and not hasattr(inner, "step"):
            inner = inner._env
        # Use wrapped env.step for correct dm_env TimeStep / action_repeat
        ts = env.step(a.astype(np.float32))
        obs_list.append(ts.observation.copy())
        rewards.append(float(ts.reward))
        qpos.append(sim.data.qpos.copy())
        # success flag on the MetaWorldEnv after step
        mw_env = env
        while hasattr(mw_env, "_env"):
            mw_env = mw_env._env
        # MetaWorldEnv.step sets reward from info['success']; last reward is enough
        if ts.last():
            break

    obs = np.stack(obs_list)
    reward = np.asarray(rewards, dtype=np.float32)
    parts = parse_obs(obs)
    orig = np.load(demo_path)
    orig_obs = orig["observation"]
    # Align lengths for RMSE
    n = min(len(obs), len(orig_obs))
    o = obs[:n]
    g = orig_obs[:n]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "seed": int(seed),
        "replay_T": int(len(reward)),
        "replay_last_reward": float(reward[-1]),
        "replay_success": bool(reward[-1] > 0),
        "replay_reward_sum": float(reward.sum()),
        "aligned_T": int(n),
        "rmse_hand": rmse(o[:, 0:3], g[:, 0:3]),
        "rmse_hammer": rmse(o[:, 4:7], g[:, 4:7]),
        "rmse_nail": rmse(o[:, 11:14], g[:, 11:14]),
        "rmse_full_obs": rmse(o, g),
        "replay_hand_last": parts["hand_pos"][-1].tolist(),
        "replay_hammer_last": parts["hammer_pos"][-1].tolist(),
        "replay_nail_last": parts["nail_pos"][-1].tolist(),
        "replay_dist_nail_goal_last": float(
            np.linalg.norm(parts["nail_pos"][-1] - parts["goal_pos"][-1])
        ),
        "qpos": np.stack(qpos),
    }


def render_last_qpos_frame(qpos_npz: Path, out_png: Path) -> None:
    """mujoco 3.x only — no mujoco_py."""
    import imageio
    import mujoco

    xml = (
        "/home/ktang/anaconda3/envs/cs224r-hw2-local/lib/python3.10/site-packages/"
        "metaworld/envs/assets_v2/sawyer_xyz/sawyer_hammer.xml"
    )
    q = np.load(qpos_npz)["qpos"]
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    nq = min(model.nq, q.shape[1])
    renderer = mujoco.Renderer(model, height=256, width=256)
    try:
        data.qpos[:nq] = q[-1, :nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="corner")
        img = np.ascontiguousarray(renderer.render())
    finally:
        renderer.close()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_png), img)


def write_report(out: Path, original: dict, replay: dict) -> None:
    lines = [
        "# Expert demo debug: original npz vs open-loop replay",
        "",
        f"- demo: `{original['path']}`",
        f"- replay seed: `{replay['seed']}`",
        "",
        "## Original npz",
        "",
        f"- T={original['T']}, demo_success={original['demo_success']}",
        f"- reward_sum={original['reward_sum']}, n_pos={original['n_pos_reward']}, last={original['reward_last']}",
        f"- |a|>1 frac={original['action_abs_gt1_frac']:.3f}, action range=[{original['action_min']:.3f}, {original['action_max']:.3f}]",
        f"- nail0={original['nail0']} → nail_last={original['nail_last']} (goal={original['goal']})",
        f"- dist nail→goal: {original['dist_nail_goal_0']:.4f} → {original['dist_nail_goal_last']:.4f}",
        f"- dist hammer→goal last={original['dist_hammer_goal_last']:.4f}, hand→hammer last={original['dist_hand_hammer_last']:.4f}",
        "",
        "## Open-loop replay (same pipeline as dump_expert_progress mp4)",
        "",
        f"- replay_T={replay['replay_T']}, **replay_success={replay['replay_success']}**",
        f"- replay_last_reward={replay['replay_last_reward']}, reward_sum={replay['replay_reward_sum']}",
        f"- replay nail_last={replay['replay_nail_last']}, dist nail→goal last={replay['replay_dist_nail_goal_last']:.4f}",
        "",
        "## Obs mismatch (original vs replay, aligned length)",
        "",
        f"- aligned_T={replay['aligned_T']}",
        f"- RMSE hand={replay['rmse_hand']:.4f}, hammer={replay['rmse_hammer']:.4f}, nail={replay['rmse_nail']:.4f}, full={replay['rmse_full_obs']:.4f}",
        "",
        "## Interpretation",
        "",
        "- `index.md` / plot use **original** demo_success.",
        "- Expert **mp4** uses open-loop replay qpos; its visual outcome is **replay_success**, which can differ.",
        "- Large RMSE means the video is not the recorded expert states.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"original": original, "replay": {k: v for k, v in replay.items() if k != "qpos"}}
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    original = summarize_original(args.demo)
    replay = open_loop_replay(args.demo, args.seed)

    # Save this replay's qpos for inspection
    np.savez_compressed(
        args.out / "open_loop_replay_qpos.npz",
        qpos=replay["qpos"],
        replay_success=np.array(replay["replay_success"]),
        seed=np.array(args.seed),
    )

    # Plot original obs trajectory
    from visualize_trajectories import load_demo, plot_trajectory

    demo = load_demo(args.demo)
    plot_trajectory(
        demo,
        args.out / "original_trajectory_plot.png",
        title=f"ORIGINAL {args.demo.stem} demo_success={original['demo_success']}",
        mark_clip=True,
    )

    # Last frame from already-saved expert qpos (what the shipped mp4 used)
    if SAVED_QPOS.exists():
        # render without mujoco_py on LD path pollution
        old_ld = os.environ.pop("LD_LIBRARY_PATH", None)
        try:
            render_last_qpos_frame(SAVED_QPOS, args.out / "saved_expert_qpos_last_frame.png")
        finally:
            if old_ld is not None:
                os.environ["LD_LIBRARY_PATH"] = old_ld
        print("wrote", args.out / "saved_expert_qpos_last_frame.png")

    # Last frame from this debug replay
    np.savez_compressed(args.out / "_tmp_render_qpos.npz", qpos=replay["qpos"])
    old_ld = os.environ.pop("LD_LIBRARY_PATH", None)
    try:
        render_last_qpos_frame(args.out / "_tmp_render_qpos.npz", args.out / "replay_last_frame.png")
    finally:
        if old_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = old_ld
        (args.out / "_tmp_render_qpos.npz").unlink(missing_ok=True)

    write_report(args.out, original, replay)
    print("wrote", args.out / "report.md")


if __name__ == "__main__":
    main()
