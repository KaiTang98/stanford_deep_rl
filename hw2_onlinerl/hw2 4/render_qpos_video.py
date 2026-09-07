#!/usr/bin/env python3
"""Render saved qpos with DeepMind mujoco 3.x (not mujoco_py).

Do not import metaworld/mujoco_py here — those backends segfault on this box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

XML = (
    "/home/ktang/anaconda3/envs/cs224r-hw2-local/lib/python3.10/site-packages/"
    "metaworld/envs/assets_v2/sawyer_xyz/sawyer_hammer.xml"
)
ROOT = Path(__file__).resolve().parent
QPOS_DIR = ROOT / "viz" / "qpos"
OUT_DIR = ROOT / "viz"


def render_npz(npz_path: Path, camera: str = "corner", size: int = 256, fps: int = 20) -> Path:
    data_in = np.load(npz_path)
    qpos = data_in["qpos"]
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    nq = min(model.nq, qpos.shape[1])
    renderer = mujoco.Renderer(model, height=size, width=size)
    frames = []
    try:
        for q in qpos:
            data.qpos[:nq] = q[:nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(np.ascontiguousarray(renderer.render()))
    finally:
        renderer.close()
    out = OUT_DIR / f"{npz_path.stem}.mp4"
    imageio.mimsave(str(out), frames, fps=fps)
    print(f"wrote {out} frames={len(frames)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="corner")
    args = parser.parse_args()
    files = sorted(QPOS_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"no qpos dumps in {QPOS_DIR}")
    for p in files:
        render_npz(p, camera=args.camera)


if __name__ == "__main__":
    main()
