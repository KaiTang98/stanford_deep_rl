#!/usr/bin/env python3
"""Write expert demo plots + RGB videos into viz/retrain_process/{expert,ppo,3a,3b}/expert."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import mw
from collect_qpos import get_sim
from visualize_trajectories import DEMO_DIR, load_demo, pick_demo_paths, plot_trajectory
from viz_progress import DEFAULT_ROOT, render_qpos_mp4

ROOT = Path(__file__).resolve().parent


def replay_demo_qpos(env, demo: dict):
    sim = get_sim(env)
    env.reset()
    qs = [sim.data.qpos.copy()]
    for a in demo["action"]:
        ts = env.step(a.astype(np.float32))
        qs.append(sim.data.qpos.copy())
        if ts.last():
            break
    return np.stack(qs)


def main():
    dest_root = DEFAULT_ROOT
    expert_dir = dest_root / "expert"
    expert_dir.mkdir(parents=True, exist_ok=True)
    env = mw.make()
    paths = pick_demo_paths(DEMO_DIR, n=3)
    index = ["# Expert demos (short / mid / long)", ""]
    for p in paths:
        demo = load_demo(p)
        plot_trajectory(
            demo,
            expert_dir / f"{demo['name']}.png",
            title=f"expert {demo['name']} success={demo['success']}",
            mark_clip=True,
        )
        qpos = replay_demo_qpos(env, demo)
        npz = expert_dir / f"{demo['name']}_qpos.npz"
        np.savez_compressed(npz, qpos=qpos, success=np.array(demo["success"]))
        render_qpos_mp4(npz, expert_dir / f"{demo['name']}.mp4")
        index.append(f"- {demo['name']}: T={demo['T']} success={demo['success']}")
    (expert_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    for method in ("ppo", "3a", "3b"):
        target = dest_root / method / "expert"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(expert_dir, target)
        print("copied expert ->", target)
    (dest_root / "README.md").write_text(
        "\n".join(
            [
                "# Retrain process visualizations",
                "",
                "Each method folder (`ppo`, `3a`, `3b`) contains:",
                "- `expert/`: the same 3 BC demos (plot + mp4)",
                "- `pctXXX_frameYYYYYYY/`: 3 random eval rollouts every 5% of training frames",
                "",
                "RGB is rendered with mujoco 3.x from saved qpos (mujoco_py render segfaults).",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
