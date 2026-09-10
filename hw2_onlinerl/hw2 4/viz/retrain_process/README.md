# Retrain process visualizations

Each method folder (`ppo`, `3a`, `3b`) contains:
- `expert/`: the same 3 BC demos (plot + mp4)
- `pctXXX_frameYYYYYYY/`: 3 random eval rollouts every 5% of training frames

RGB is rendered with mujoco 3.x from saved qpos (mujoco_py render segfaults).
