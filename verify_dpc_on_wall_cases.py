"""Sanity check: run DPC on the SAME 13 saved wall-contact x0 (from
wall_contact_cases.pt) and see if it reproduces the expected behavior (theta
successfully stabilized, wall only lightly touched) that we know happened in
the original 200-sample eval (98.5% theta-success but 6.5%/13 wall-contact --
meaning most of these 13 cases DID succeed on theta despite touching the
wall). If DPC reproduces mild success here, the saved x0 data is fine and the
MPC catastrophic divergence (350-800+ deg) points to a bug in the MPC path,
not genuine physical impossibility.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_ode import make_ground_truth_integrator
from dpc_policy_balance import CartPolePolicy, F_MAX, Ts, TRACK_LIMIT
from eval_balance_policy import closed_loop_rollout_ground_truth

CKPT = "policy_balance.pth"

if __name__ == "__main__":
    data = torch.load("wall_contact_cases.pt")
    x0_batch = data["x0"]
    orig_indices = data["indices"]

    policy = CartPolePolicy(nx=4, F_max=F_MAX)
    policy.load_state_dict(torch.load(CKPT, map_location="cpu"))
    policy.eval()

    integrator = make_ground_truth_integrator(Ts=Ts)
    states, controls = closed_loop_rollout_ground_truth(policy, integrator, x0_batch, T=250, track_limit=None)

    pos = states[:, :, 0]
    theta = states[:, :, 2]

    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    theta_wrapped = wrap_to_pi(theta)
    hit_wall = (pos.abs() >= TRACK_LIMIT - 1e-3).any(dim=1)
    tail = states[:, int(250 * 0.8):, :]
    tail_theta_wrapped = wrap_to_pi(tail[:, :, 2])
    success_raw = (tail[:, :, 2].abs() < 0.3).all(dim=1)
    success_wrapped = (tail_theta_wrapped.abs() < 0.3).all(dim=1)

    print("[DPC로 저장된 x0 13개 재검증]")
    for i in range(len(x0_batch)):
        print(f"  idx={orig_indices[i].item():3d}: x0={x0_batch[i].numpy().round(3)}, "
              f"벽접촉={'예' if hit_wall[i] else '아니오'}, "
              f"성공(raw)={'예' if success_raw[i] else '아니오'}, "
              f"성공(wrapped)={'예' if success_wrapped[i] else '아니오'}, "
              f"final|theta|(raw)={theta[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|theta|(wrapped)={theta_wrapped[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|pos|={pos[i,-1].abs().item():.3f}m, max|pos|={pos[i].abs().max().item():.3f}m")
    print(f"\n  벽 접촉: {hit_wall.sum().item()}/{len(x0_batch)}, "
          f"성공(raw): {success_raw.sum().item()}/{len(x0_batch)}, "
          f"성공(wrapped): {success_wrapped.sum().item()}/{len(x0_batch)}")

    # ---------- Plots ----------
    T = 250
    t = np.arange(T + 1) * Ts
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i in range(len(x0_batch)):
        label = f"idx={orig_indices[i].item()}"
        axes[0, 0].plot(t, theta[i].numpy(), label=label)
        axes[0, 1].plot(t, pos[i].numpy(), label=label)
        axes[1, 1].plot(np.arange(T) * Ts, controls[i, :, 0].numpy(), label=label)

    axes[0, 0].axhline(0, color="black", ls="--", lw=1)
    axes[0, 0].set_title("theta over time (13 wall-contact cases, DPC)")
    axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("theta [rad]")
    axes[0, 0].legend(fontsize=7, ncol=2)

    axes[0, 1].axhline(TRACK_LIMIT, color="red", ls="--", lw=1, label="track limit")
    axes[0, 1].axhline(-TRACK_LIMIT, color="red", ls="--", lw=1)
    axes[0, 1].set_title("cart_pos over time")
    axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("x [m]")
    axes[0, 1].legend(fontsize=7, ncol=2)

    axes[1, 0].plot(t, theta_wrapped.mean(dim=0).numpy(), label="mean |theta| (wrapped)")
    axes[1, 0].plot(t, theta_wrapped.abs().numpy().T, alpha=0.15, color="gray")
    axes[1, 0].set_title("|theta| (wrapped) over time, all 13")
    axes[1, 0].set_xlabel("time [s]"); axes[1, 0].set_ylabel("|theta| wrapped [rad]")

    axes[1, 1].axhline(F_MAX, color="red", ls="--", lw=1, label="F_max")
    axes[1, 1].axhline(-F_MAX, color="red", ls="--", lw=1)
    axes[1, 1].set_title("Control input u")
    axes[1, 1].set_xlabel("time [s]"); axes[1, 1].set_ylabel("u [N]")
    axes[1, 1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig("dpc_wall_cases_trajectories.png", dpi=130)
    print("\nSaved plot to dpc_wall_cases_trajectories.png")
