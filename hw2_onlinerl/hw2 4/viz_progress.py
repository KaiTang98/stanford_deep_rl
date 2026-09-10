#!/usr/bin/env python3
"""Dump 3 random eval rollouts (plots + RGB via mujoco 3.x) at training milestones.

Does not call mujoco_py sim.render (segfaults on this box). Saves qpos, then
renders in a subprocess with DeepMind mujoco.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from visualize_trajectories import parse_obs, plot_trajectory, unwrap_sim

ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "viz" / "retrain_process"
RENDER_SCRIPT = ROOT / "render_qpos_video.py"


def checkpoint_frames(num_train_frames: int, action_repeat: int, pct_step: float = 0.05):
    n = int(round(1.0 / pct_step))
    frames = []
    for i in range(n + 1):
        f = int(round(i * pct_step * num_train_frames))
        f = (f // action_repeat) * action_repeat
        frames.append(f)
    last = (num_train_frames // action_repeat - 1) * action_repeat
    if last > 0:
        frames.append(last)
    return sorted(set(int(x) for x in frames if x >= 0))


def get_sim(env):
    inner = unwrap_sim(env)
    return inner.sim if hasattr(inner, "sim") else inner._env.sim


def rollout_eval(agent, env, seed: int):
    """Deterministic policy, random reset seed. Collect obs/action/qpos."""
    np.random.seed(int(seed) % (2**31 - 1))
    torch.manual_seed(int(seed) % (2**31 - 1))
    sim = get_sim(env)
    ts = env.reset()
    obs_list = [ts.observation.copy()]
    act_list = [np.zeros(4, dtype=np.float32)]
    rewards = [0.0]
    qpos = [sim.data.qpos.copy()]
    agent.train(False)
    while not ts.last():
        with torch.no_grad():
            action = agent.act(ts.observation, eval_mode=True)
        ts = env.step(action)
        obs_list.append(ts.observation.copy())
        act_list.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(ts.reward))
        qpos.append(sim.data.qpos.copy())
    obs = np.stack(obs_list)
    action = np.stack(act_list)
    reward = np.asarray(rewards, dtype=np.float32)
    parts = parse_obs(obs)
    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "success": bool(reward[-1] > 0),
        "T": len(reward),
        "qpos": np.stack(qpos),
        **parts,
    }


def render_qpos_mp4(qpos_npz: Path, out_mp4: Path) -> None:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    # Avoid Mesa/Isaac LD_LIBRARY_PATH mixing in the renderer process.
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CPATH", None)
    env.pop("LIBRARY_PATH", None)
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--npz",
        str(qpos_npz),
        "--out",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=False, env=env, cwd=str(ROOT))


class ProgressDumper:
    def __init__(self, out_dir: Path, num_train_frames: int, action_repeat: int, seed: int, n_rollouts: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames = checkpoint_frames(num_train_frames, action_repeat)
        self.frame_set = set(self.frames)
        self.done = set()
        self.seed = int(seed)
        self.n_rollouts = n_rollouts
        self.num_train_frames = num_train_frames
        (self.out_dir / "checkpoints.txt").write_text(
            "\n".join(str(f) for f in self.frames) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_workspace(cls, workspace, method: str):
        root = Path(os.environ.get("HW2_PROGRESS_VIZ_ROOT", str(DEFAULT_ROOT)))
        out = root / method
        return cls(
            out,
            num_train_frames=int(workspace.cfg.num_train_frames),
            action_repeat=int(workspace.cfg.action_repeat),
            seed=int(workspace.cfg.seed),
        )

    def maybe_dump(self, workspace, force: bool = False) -> None:
        frame = int(workspace.global_frame)
        if not force and frame not in self.frame_set:
            return
        if frame in self.done:
            return
        self.dump(workspace, frame)
        self.done.add(frame)

    def dump(self, workspace, frame: int) -> None:
        pct = 100.0 * frame / max(self.num_train_frames, 1)
        folder = self.out_dir / f"pct{pct:05.1f}_frame{frame:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState((self.seed * 10007 + frame) % (2**31 - 1))
        seeds = rng.randint(0, 10**9, size=self.n_rollouts)
        lines = [
            f"frame={frame} step={workspace.global_step} episode={workspace.global_episode}",
            f"pct≈{pct:.1f}% of {self.num_train_frames} frames",
            f"seeds={list(map(int, seeds))}",
            "",
        ]
        print(f"[progress_viz] {self.out_dir.name} frame={frame} pct={pct:.1f} -> {folder}")
        for i, seed in enumerate(seeds):
            traj = rollout_eval(workspace.agent, workspace.eval_env, int(seed))
            tag = "success" if traj["success"] else "fail"
            stem = f"rollout{i}_{tag}_seed{int(seed)}"
            plot_trajectory(
                traj,
                folder / f"{stem}.png",
                title=f"{self.out_dir.name}  frame={frame}  ({pct:.0f}%)  {stem}",
                mark_clip=False,
            )
            npz = folder / f"{stem}_qpos.npz"
            np.savez_compressed(
                npz,
                qpos=traj["qpos"],
                success=np.array(traj["success"]),
                seed=np.array(int(seed)),
                frame=np.array(frame),
            )
            render_qpos_mp4(npz, folder / f"{stem}.mp4")
            lines.append(f"{stem}  T={traj['T']}  last_reward={traj['reward'][-1]:.3f}")
        (folder / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
