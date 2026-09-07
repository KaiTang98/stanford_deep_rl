#!/usr/bin/env python3
"""Collect qpos from mujoco_py physics (no sim.render) for later RGB replay.

Needs the HW2 env.sh so mujoco_py can import. Do not call env.render().
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import numpy as np
import torch

import mw
from visualize_trajectories import (
    DEMO_DIR,
    SNAPSHOT_3A,
    SNAPSHOT_3B,
    load_agent,
    load_demo,
    pick_demo_paths,
    unwrap_sim,
)

ROOT = Path(__file__).resolve().parent
QPOS_DIR = ROOT / "viz" / "qpos"


def get_sim(env):
    inner = unwrap_sim(env)
    return inner.sim if hasattr(inner, "sim") else inner._env.sim


def dump(path: Path, qpos: np.ndarray, success: bool, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=qpos, success=np.array(success), kind=kind)
    print(f"wrote {path} T={len(qpos)} nq={qpos.shape[1]} success={success}")


def collect_policy(label: str, snapshot: Path, env, device, seeds) -> None:
    agent, _ = load_agent(snapshot, device)
    sim = get_sim(env)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        ts = env.reset()
        qs = [sim.data.qpos.copy()]
        agent.train(False)
        while not ts.last():
            with torch.no_grad():
                action = agent.act(ts.observation, eval_mode=True)
            ts = env.step(action)
            qs.append(sim.data.qpos.copy())
        success = bool(ts.reward > 0)
        tag = "success" if success else "fail"
        dump(QPOS_DIR / f"{label}_seed{seed}_{tag}.npz", np.stack(qs), success, label)


def collect_demo(env, path: Path) -> None:
    demo = load_demo(path)
    sim = get_sim(env)
    env.reset()
    qs = [sim.data.qpos.copy()]
    # Replay stored actions; env clips to [-1, 1]. Reset noise means this is
    # not a bit-exact reconstruction of the recorded demo states.
    actions = demo["action"]
    for a in actions:
        ts = env.step(a.astype(np.float32))
        qs.append(sim.data.qpos.copy())
        if ts.last():
            break
    dump(
        QPOS_DIR / f"demo_{demo['name']}_replay.npz",
        np.stack(qs),
        bool(demo["success"]),
        "demo_action_replay",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    env = mw.make()

    collect_demo(env, pick_demo_paths(DEMO_DIR, n=3)[-1])  # longest
    collect_policy("3A", SNAPSHOT_3A, env, device, seeds=(100, 103))
    collect_policy("3B", SNAPSHOT_3B, env, device, seeds=(100, 103))


if __name__ == "__main__":
    main()
