"""Evaluate the fully-observable NSSM (no observer -- transition model
accuracy only). Same physical-unit metrics as eval_sysid.py (short/long
horizon RMSE/MAE, angle-wrapped), but starting from the TRUE x0 directly
(no observer estimation error mixed in) -- isolates the transition model's
own accuracy.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_ode import make_ground_truth_integrator, rollout as gt_rollout
from generate_dataset import generate_prbs
from train_nssm_sysid_fullobs import build_fullobs_sysid_problem


def wrap_angle_diff(diff):
    return (diff + np.pi) % (2 * np.pi) - np.pi


def load_model(ckpt_path, N_pred=100, Ts=0.02, transition_hsizes=[32, 32], integration="euler"):
    problem = build_fullobs_sysid_problem(N_pred=N_pred, Ts=Ts, transition_hsizes=transition_hsizes,
                                           integration=integration)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    problem.load_state_dict(state_dict)
    problem.eval()
    # nodes = [dynamics_system, fy_node]; dynamics_system.nodes[0] = nssm_node -> .callable = transition
    transition = problem.nodes[0].nodes[0].callable
    return transition


@torch.no_grad()
def identified_rollout(transition, x0, u_seq):
    """x0: (batch, 4) true initial state. u_seq: (batch, T, 1)."""
    batch, T, _ = u_seq.shape
    states = torch.zeros(batch, T + 1, x0.shape[-1])
    states[:, 0] = x0
    x = x0
    for k in range(T):
        x = transition(x, u_seq[:, k])
        states[:, k + 1] = x
    return states


@torch.no_grad()
def evaluate(ckpt_path, N_pred=100, Ts=0.02, F_max=5.0, num_test=200,
             long_horizon_steps=150, seed=999, transition_hsizes=[32, 32], integration="euler"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transition = load_model(ckpt_path, N_pred, Ts, transition_hsizes, integration)
    integrator = make_ground_truth_integrator(Ts=Ts)

    # ---------- Short-horizon ----------
    p0 = torch.FloatTensor(num_test, 1).uniform_(-1.0, 1.0)
    p_dot0 = torch.FloatTensor(num_test, 1).uniform_(-1.0, 1.0)
    theta0 = torch.FloatTensor(num_test, 1).uniform_(-np.pi, np.pi)
    theta_dot0 = torch.FloatTensor(num_test, 1).uniform_(-6.0, 6.0)
    x0_true = torch.cat([p0, p_dot0, theta0, theta_dot0], dim=-1)

    u_seq_short = generate_prbs(num_test, N_pred, Ts, F_max=F_max, min_hold_time=0.1, max_hold_time=0.5)
    x_target = gt_rollout(integrator, x0_true, u_seq_short)  # (batch, N_pred+1, 4)

    x_pred = identified_rollout(transition, x0_true, u_seq_short)

    theta_err_raw = (x_pred[:, :, 2] - x_target[:, :, 2])
    theta_err_wrapped = wrap_angle_diff(theta_err_raw)

    print("=" * 60)
    print(f"[Short-horizon (fully observable): {N_pred} steps = {N_pred*Ts:.2f}s]")
    print(f"  cart_pos   RMSE: {(x_pred[:,:,0]-x_target[:,:,0]).pow(2).mean().sqrt().item():.4f} m")
    print(f"  cart_vel   RMSE: {(x_pred[:,:,1]-x_target[:,:,1]).pow(2).mean().sqrt().item():.4f} m/s")
    theta_rmse_raw = theta_err_raw.pow(2).mean().sqrt().item()
    theta_rmse_wrapped = theta_err_wrapped.pow(2).mean().sqrt().item()
    print(f"  theta      RMSE: {theta_rmse_raw:.4f} rad  ({np.degrees(theta_rmse_raw):.2f} deg)")
    print(f"  theta      RMSE (angle-wrapped): {theta_rmse_wrapped:.4f} rad ({np.degrees(theta_rmse_wrapped):.2f} deg)  <- compare to raw above")
    print(f"  theta_dot  RMSE: {(x_pred[:,:,3]-x_target[:,:,3]).pow(2).mean().sqrt().item():.4f} rad/s")

    # ---------- Long-horizon ----------
    T_long = long_horizon_steps
    u_seq_long = generate_prbs(num_test, T_long, Ts, F_max=F_max, min_hold_time=0.1, max_hold_time=0.5)
    x_target_long = gt_rollout(integrator, x0_true, u_seq_long)
    x_pred_long = identified_rollout(transition, x0_true, u_seq_long)

    theta_err_long_raw = (x_pred_long[:, :, 2] - x_target_long[:, :, 2])
    theta_err_long_wrapped = wrap_angle_diff(theta_err_long_raw)

    print("=" * 60)
    print(f"[Long-horizon (fully observable): {T_long} steps = {T_long*Ts:.2f}s, {T_long/N_pred:.1f}x training horizon]")
    print(f"  cart_pos MAE at step {N_pred} (train horizon): {(x_pred_long[:,N_pred,0]-x_target_long[:,N_pred,0]).abs().mean().item():.4f} m")
    print(f"  cart_pos MAE at final step {T_long}: {(x_pred_long[:,-1,0]-x_target_long[:,-1,0]).abs().mean().item():.4f} m")
    theta_mae_at_train_raw = theta_err_long_raw[:, N_pred].abs().mean().item()
    theta_mae_at_train_wrapped = theta_err_long_wrapped[:, N_pred].abs().mean().item()
    print(f"  theta    MAE at step {N_pred} (train horizon): {theta_mae_at_train_raw:.4f} rad ({np.degrees(theta_mae_at_train_raw):.2f} deg)")
    print(f"  theta    MAE at step {N_pred} (angle-wrapped): {theta_mae_at_train_wrapped:.4f} rad ({np.degrees(theta_mae_at_train_wrapped):.2f} deg)  <- compare to raw above")
    theta_mae_final_raw = theta_err_long_raw[:, -1].abs().mean().item()
    theta_mae_final_wrapped = theta_err_long_wrapped[:, -1].abs().mean().item()
    print(f"  theta    MAE at final step {T_long}: {theta_mae_final_raw:.4f} rad ({np.degrees(theta_mae_final_raw):.2f} deg)")
    print(f"  theta    MAE at final step {T_long} (angle-wrapped): {theta_mae_final_wrapped:.4f} rad ({np.degrees(theta_mae_final_wrapped):.2f} deg)  <- compare to raw above")

    # ---------- Plot ----------
    t_long = np.arange(T_long + 1) * Ts
    mean_pos_err = (x_pred_long[:, :, 0] - x_target_long[:, :, 0]).abs().mean(dim=0).numpy()
    mean_theta_err_raw = theta_err_long_raw.abs().mean(dim=0).numpy()
    mean_theta_err_wrapped = theta_err_long_wrapped.abs().mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t_long, mean_pos_err)
    axes[0].axvline(N_pred * Ts, color="red", ls="--", label="training horizon")
    axes[0].set_title("Mean |cart_pos error| over long rollout (fully observable)")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("error [m]")
    axes[0].legend()

    axes[1].plot(t_long, np.degrees(mean_theta_err_raw), label="raw")
    axes[1].plot(t_long, np.degrees(mean_theta_err_wrapped), label="angle-wrapped")
    axes[1].axvline(N_pred * Ts, color="red", ls="--", label="training horizon")
    axes[1].set_title("Mean |theta error| over long rollout (fully observable)")
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("error [deg]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("sysid_eval_fullobs.png", dpi=130)
    print("\nSaved plot to sysid_eval_fullobs.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="sysid.pth")
    parser.add_argument("--N_pred", type=int, default=100)
    parser.add_argument("--Ts", type=float, default=0.02)
    parser.add_argument("--F_max", type=float, default=5.0)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--long_horizon_steps", type=int, default=150)
    parser.add_argument("--hsizes", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--integration", type=str, default="euler", choices=["euler", "rk4"])
    args = parser.parse_args()

    evaluate(args.ckpt, N_pred=args.N_pred, Ts=args.Ts, F_max=args.F_max,
              num_test=args.num_test, long_horizon_steps=args.long_horizon_steps,
              transition_hsizes=args.hsizes, integration=args.integration)
