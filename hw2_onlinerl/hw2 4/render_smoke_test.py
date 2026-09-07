#!/usr/bin/env python3
"""Independent smoke test for mujoco_py offscreen render.

Run in a *separate* process from training. Does not write ~/.bashrc.

Priority (from the viz plan):
  1) Single GPU + NVIDIA EGL, without user-dir Mesa on LD_LIBRARY_PATH
  2) xvfb + glfw if available
  3) Report failure (caller falls back to trajectory plots)

Example:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \\
    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia \\
    python render_smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def try_render(camera: str = "corner", size: int = 84) -> bool:
    # Import after env vars are set by the caller / this script.
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    import mw

    env = mw.make()
    env.reset()
    # Walk to Sawyer sim
    e = env
    while hasattr(e, "_env"):
        e = e._env
    sim = e._env.sim if hasattr(e, "_env") else e.sim
    print(
        "MUJOCO_GL=",
        os.environ.get("MUJOCO_GL"),
        "CUDA_VISIBLE_DEVICES=",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MUJOCO_EGL_DEVICE_ID=",
        os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "LD_LIBRARY_PATH=",
        os.environ.get("LD_LIBRARY_PATH", "")[:200],
    )
    img = sim.render(width=size, height=size, mode="offscreen", camera_name=camera)
    print("render ok shape=", getattr(img, "shape", None), "dtype=", getattr(img, "dtype", None))
    out = Path(__file__).resolve().parent / "viz" / "render_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        # mujoco_py returns upside-down RGB sometimes; flip for viewing
        frame = img[::-1] if img.ndim == 3 else img
        imageio.imwrite(str(out), frame)
        print("wrote", out)
    except Exception as exc:
        print("save skipped:", exc)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--size", type=int, default=84)
    args = parser.parse_args()
    try:
        ok = try_render(camera=args.camera, size=args.size)
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
