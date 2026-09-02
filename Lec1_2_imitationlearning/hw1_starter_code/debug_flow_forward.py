"""Entry point for debugging one Flow Matching forward pass.

Set breakpoints in the real modules, then run this file in the debugger:

    networks.py  TemporalNoisePredictor.forward
    networks.py  ConditionalUnet1D.forward
    networks.py  ConditionalResidualBlock1D.forward
    networks.py  FlowMatchingPolicy.forward

Cursor / VS Code: open this file → Run and Debug → Python Debugger: Current File
Terminal:  python -m pdb debug_flow_forward.py
"""

import torch

from networks import FlowMatchingPolicy


def main():
    torch.manual_seed(0)
    B, T, C, S = 8, 20, 1, 4

    policy = FlowMatchingPolicy(
        state_dim=S, pred_horizon=T, action_dim=C, num_steps=20, device="cpu"
    )
    policy.eval()

    # Fake what interpolate() would produce (your TODO is not called here).
    a_clean = torch.rand(B, T)
    noise = torch.randn(B, T)
    tau = torch.rand(B)
    a_tau = tau.view(B, 1) * a_clean + (1.0 - tau.view(B, 1)) * noise
    state = torch.tensor([[0.40, 0.35, 0.65, 0.50]]).repeat(B, 1)

    # Step INTO this call: it is FlowMatchingPolicy.forward → TemporalNoisePredictor
    velocity = policy(a_tau, state, tau)

    return velocity


if __name__ == "__main__":
    main()
