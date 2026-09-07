#!/usr/bin/env python3
"""Visualize BC demos and learned off-policy policies (3A / 3B).

Observation (39-d, Meta-World hammer-v2 V2 frame-stack):
  obs[0:3]   hand / end-effector XYZ
  obs[3]     gripper aperture in [0, 1] (1 = open)
  obs[4:7]   hammer XYZ
  obs[7:11]  hammer quaternion
  obs[11:14] nail XYZ
  obs[14:18] nail quaternion
  obs[18:36] previous-frame copy of the 18-d block above
  obs[36:39] goal XYZ (visible; mw.py sets partially_observable=False)

Action (4-d):
  action[0:3] end-effector XYZ delta (env clips to [-1, 1], then * 0.01 m)
  action[3]   gripper effort (sim applies [a3, -a3] to the fingers)

PPO note: the finished PPO run used save_snapshot=false, so there is no
snapshot.pt — this script cannot replay the PPO policy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Physical step only — never call env.render() here.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mw
from off_policy import ACAgent  # noqa: F401 — needed for torch.load unpickle

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demos"
VIZ_DIR = ROOT / "viz"

SNAPSHOT_3A = (
    ROOT
    / "Logdir"
    / "run_150227_device=cuda,save_snapshot=true,save_video=false"
    / "snapshot.pt"
)
SNAPSHOT_3B = (
    ROOT
    / "Logdir"
    / "run_154425_agent.num_critics=10,device=cuda,save_snapshot=true,save_video=false,utd=5"
    / "snapshot.pt"
)

# Named slices for the 39-d observation.
OBS_SLICES = {
    "hand_pos": slice(0, 3),
    "gripper": slice(3, 4),
    "hammer_pos": slice(4, 7),
    "hammer_quat": slice(7, 11),
    "nail_pos": slice(11, 14),
    "nail_quat": slice(14, 18),
    "prev_frame": slice(18, 36),
    "goal_pos": slice(36, 39),
}


def parse_obs(obs: np.ndarray) -> dict:
    """Split a (T, 39) or (39,) observation into named parts."""
    o = np.asarray(obs, dtype=np.float32)
    if o.ndim == 1:
        o = o[None, :]
    return {name: o[:, sl] for name, sl in OBS_SLICES.items()}


def load_demo(path: Path) -> dict:
    data = np.load(path)
    obs = data["observation"]
    action = data["action"]
    reward = data["reward"].reshape(-1)
    # Skip the dummy FIRST timestep action (zeros / garbage first row).
    parts = parse_obs(obs)
    return {
        "name": path.stem,
        "obs": obs,
        "action": action,
        "reward": reward,
        "success": bool(reward[-1] > 0),
        "T": len(reward),
        **parts,
    }


def pick_demo_paths(demo_dir: Path, n: int = 3) -> list[Path]:
    """Pick short / medium / long demos by episode length in the filename."""
    paths = sorted(demo_dir.glob("demo_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no demos in {demo_dir}")
    lengths = []
    for p in paths:
        # filenames look like demo_8_46.npz → length 46
        try:
            lengths.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            lengths.append((len(np.load(p)["reward"]), p))
    lengths.sort(key=lambda x: x[0])
    if len(lengths) <= n:
        return [p for _, p in lengths]
    idxs = [0, len(lengths) // 2, len(lengths) - 1]
    return [lengths[i][1] for i in idxs[:n]]


def summarize_demo_actions(demo_dir: Path) -> str:
    paths = sorted(demo_dir.glob("demo_*.npz"))
    acts = []
    for p in paths:
        a = np.load(p)["action"][1:]  # skip dummy first
        acts.append(a)
    A = np.concatenate(acts, axis=0)
    frac = (np.abs(A) > 1).any(axis=1).mean()
    return (
        f"demos={len(paths)} transitions={len(A)} "
        f"|a|>1 any-dim frac={frac:.2%} "
        f"min={A.min():.2f} max={A.max():.2f}"
    )


def plot_trajectory(
    traj: dict,
    out_path: Path,
    title: str,
    mark_clip: bool = False,
) -> None:
    """Save a 3D path + time-series figure for one trajectory."""
    hand = traj["hand_pos"]
    hammer = traj["hammer_pos"]
    nail = traj["nail_pos"]
    goal = traj["goal_pos"]
    gripper = traj["gripper"].reshape(-1)
    action = traj["action"]
    # Align action rows with obs: demos/store include FIRST dummy action at index 0.
    # For plot, use actions starting at index 1 when lengths match obs.
    if len(action) == len(hand):
        act = action.copy()
        act[0] = np.nan  # dummy FIRST step
    else:
        act = action

    T = len(hand)
    t = np.arange(T)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=12)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(hand[:, 0], hand[:, 1], hand[:, 2], label="hand", color="C0")
    ax3d.plot(hammer[:, 0], hammer[:, 1], hammer[:, 2], label="hammer", color="C1")
    ax3d.plot(nail[:, 0], nail[:, 1], nail[:, 2], label="nail", color="C2")
    ax3d.scatter(
        goal[-1, 0],
        goal[-1, 1],
        goal[-1, 2],
        c="red",
        marker="*",
        s=80,
        label="goal",
    )
    ax3d.scatter(hand[0, 0], hand[0, 1], hand[0, 2], c="C0", marker="o", s=40)
    ax3d.scatter(hand[-1, 0], hand[-1, 1], hand[-1, 2], c="C0", marker="x", s=40)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.set_title("3D trajectories")

    ax_d = fig.add_subplot(2, 2, 2)
    hand_goal = np.linalg.norm(hand - goal, axis=1)
    hammer_goal = np.linalg.norm(hammer - goal, axis=1)
    nail_goal = np.linalg.norm(nail - goal, axis=1)
    hand_hammer = np.linalg.norm(hand - hammer, axis=1)
    ax_d.plot(t, hand_goal, label="hand→goal")
    ax_d.plot(t, hammer_goal, label="hammer→goal")
    ax_d.plot(t, nail_goal, label="nail→goal")
    ax_d.plot(t, hand_hammer, label="hand→hammer", linestyle="--")
    ax_d.set_xlabel("t")
    ax_d.set_ylabel("distance")
    ax_d.legend(fontsize=8)
    ax_d.set_title("Distances")
    ax_d.grid(True, alpha=0.3)

    ax_g = fig.add_subplot(2, 2, 3)
    ax_g.plot(t, gripper, color="C4")
    ax_g.set_xlabel("t")
    ax_g.set_ylabel("gripper (0=closed, 1=open)")
    ax_g.set_ylim(-0.05, 1.05)
    ax_g.set_title("Gripper aperture")
    ax_g.grid(True, alpha=0.3)

    ax_a = fig.add_subplot(2, 2, 4)
    labels = ["dx", "dy", "dz", "grip"]
    for i in range(4):
        ax_a.plot(t[: len(act)], act[:, i], label=labels[i])
    if mark_clip:
        # Highlight timesteps where any |a| > 1 (expert demos before env clip).
        over = np.where(np.nan_to_num(np.abs(act), nan=0.0).max(axis=1) > 1.0)[0]
        if len(over):
            ax_a.scatter(
                over,
                np.zeros_like(over, dtype=float),
                c="red",
                s=12,
                zorder=5,
                label="|a|>1",
            )
    ax_a.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax_a.axhline(-1.0, color="gray", linestyle=":", linewidth=1)
    ax_a.set_xlabel("t")
    ax_a.set_ylabel("action")
    ax_a.legend(fontsize=8)
    ax_a.set_title("Actions (env clips to [-1, 1])")
    ax_a.grid(True, alpha=0.3)

    success = traj.get("success", False)
    fig.text(
        0.5,
        0.02,
        f"success={success}  T={T}  "
        f"obs=39d (hand/gripper/hammer/nail + prev + goal)  action=4d",
        ha="center",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def unwrap_sim(env):
    """Walk dm_env wrappers down to the MetaWorldEnv / Sawyer sim."""
    e = env
    while hasattr(e, "_env"):
        e = e._env
    return e


def rollout_policy(agent, env, seed: int, device: torch.device) -> dict:
    """Deterministic eval rollout; never calls render."""
    # mw.make() resets with its own hand_init noise; seed numpy for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    time_step = env.reset()
    obs_list = [time_step.observation.copy()]
    act_list = [np.zeros(4, dtype=np.float32)]  # dummy FIRST, match demo layout
    rewards = [0.0]

    agent.train(False)
    while not time_step.last():
        with torch.no_grad():
            action = agent.act(time_step.observation, eval_mode=True)
        time_step = env.step(action)
        obs_list.append(time_step.observation.copy())
        act_list.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(time_step.reward))

    obs = np.stack(obs_list, axis=0)
    action = np.stack(act_list, axis=0)
    reward = np.asarray(rewards, dtype=np.float32)
    parts = parse_obs(obs)
    return {
        "name": f"seed{seed}",
        "obs": obs,
        "action": action,
        "reward": reward,
        "success": bool(reward[-1] > 0),
        "T": len(reward),
        **parts,
    }


def load_agent(snapshot_path: Path, device: torch.device):
    payload = torch.load(snapshot_path, map_location=device)
    agent = payload["agent"]
    agent.device = device
    agent.actor.to(device)
    agent.critic.to(device)
    if hasattr(agent, "critic_target"):
        agent.critic_target.to(device)
    agent.train(False)
    step = payload.get("_global_step", "?")
    return agent, step


def write_readme(viz_dir: Path, demo_summary: str, results: dict) -> None:
    lines = [
        "# HW2 trajectory visualizations",
        "",
        "## Observation (39-d) / Action (4-d)",
        "",
        "| Slice | Meaning |",
        "|---|---|",
        "| `obs[0:3]` | hand XYZ |",
        "| `obs[3]` | gripper aperture `[0,1]` |",
        "| `obs[4:7]` | hammer XYZ |",
        "| `obs[7:11]` | hammer quat |",
        "| `obs[11:14]` | nail XYZ |",
        "| `obs[14:18]` | nail quat |",
        "| `obs[18:36]` | previous 18-d frame |",
        "| `obs[36:39]` | goal XYZ |",
        "| `action[0:3]` | XYZ delta (clipped to `[-1,1]`, ×0.01 m) |",
        "| `action[3]` | gripper effort |",
        "",
        "## PPO",
        "",
        "The finished PPO run used `save_snapshot=false`, so there is **no** "
        "`snapshot.pt`. This folder cannot replay the PPO policy. Re-train with "
        "`save_snapshot=true` if you need PPO trajectories later.",
        "",
        "## Demo action range",
        "",
        demo_summary,
        "",
        "## Rollout summary",
        "",
    ]
    for name, info in results.items():
        lines.append(f"- **{name}**: {info}")
    lines.append("")
    path = viz_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--out", type=Path, default=VIZ_DIR)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ----- demos -----
    demo_summary = summarize_demo_actions(DEMO_DIR)
    print(demo_summary)
    demo_paths = pick_demo_paths(DEMO_DIR, n=3)
    for p in demo_paths:
        traj = load_demo(p)
        plot_trajectory(
            traj,
            out / f"demo_{traj['name']}.png",
            title=f"BC demo {traj['name']}  success={traj['success']}",
            mark_clip=True,
        )

    results = {
        "PPO": "NO snapshot.pt (save_snapshot=false) — cannot replay",
        "demos": demo_summary,
    }

    # ----- 3A / 3B rollouts -----
    env = mw.make()
    for label, snap in [("3A_utd1_2critics", SNAPSHOT_3A), ("3B_utd5_10critics", SNAPSHOT_3B)]:
        if not snap.exists():
            results[label] = f"missing snapshot: {snap}"
            print(results[label])
            continue
        print(f"loading {label} from {snap}")
        agent, step = load_agent(snap, device)
        rollouts = []
        for i in range(args.num_rollouts):
            traj = rollout_policy(agent, env, seed=100 + i, device=device)
            rollouts.append(traj)
            tag = "success" if traj["success"] else "fail"
            plot_trajectory(
                traj,
                out / f"{label}_seed{100 + i}_{tag}.png",
                title=f"{label} seed={100 + i} success={traj['success']} (train_step≈{step})",
                mark_clip=False,
            )
        n_ok = sum(1 for r in rollouts if r["success"])
        results[label] = f"{n_ok}/{len(rollouts)} success over {args.num_rollouts} eval seeds"
        # Keep one success and one fail as "representative" copies if available.
        for r in rollouts:
            if r["success"]:
                plot_trajectory(
                    r,
                    out / f"{label}_representative_success.png",
                    title=f"{label} representative SUCCESS",
                )
                break
        for r in rollouts:
            if not r["success"]:
                plot_trajectory(
                    r,
                    out / f"{label}_representative_fail.png",
                    title=f"{label} representative FAIL",
                )
                break

    write_readme(out, demo_summary, results)
    print("done. figures in", out)


if __name__ == "__main__":
    main()
