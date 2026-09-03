"""Evaluate the trained balancing policy on the REAL (ground-truth) dynamics,
not the identified model it was trained on. This is the real test: does the
policy, trained purely on an imperfect learned model, actually stabilize the
real physical system?
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_ode import make_ground_truth_integrator, apply_track_limits, make_pinned_integrator
from dpc_policy_balance import CartPolePolicy, sample_balance_x0, F_MAX, TRACK_LIMIT, Ts, load_frozen_dynamics


@torch.no_grad()
def closed_loop_rollout_identified(policy, transition, x0, T):
    """Roll out policy + the IDENTIFIED (frozen) dynamics model in closed loop
    -- i.e. exactly what the policy was optimized against during training.
    Used to isolate 'policy training itself is broken' from 'sim-to-real gap
    between identified model and ground truth'.
    x0: (batch, 4)
    """
    batch = x0.shape[0]
    states = torch.zeros(batch, T + 1, 4)
    controls = torch.zeros(batch, T, 1)
    states[:, 0] = x0
    x = x0
    for k in range(T):
        u = policy(x)
        controls[:, k] = u
        x = transition(x, u)
        states[:, k + 1] = x
    return states, controls


@torch.no_grad()
def closed_loop_rollout_ground_truth(policy, integrator, x0, T, track_limit=TRACK_LIMIT,
                                      use_wall_physics=True):
    """Roll out policy + REAL ground-truth dynamics in closed loop.
    x0: (batch, 4)
    returns states (batch, T+1, 4), controls (batch, T, 1)

    NOTE: applies apply_track_limits after every step, matching
    cartpole_ode.rollout()'s behavior -- the ground-truth physical rig has a
    hard wall at +-track_limit. This loop does its own manual stepping (rather
    than calling rollout() directly) because of the policy interaction each
    step, so the wall clipping has to be applied explicitly here too. Missing
    this meant all prior ground-truth evaluations ran on an unbounded
    (walless) track.

    use_wall_physics (default True): when pinned against the wall and pushed
    further into it, use the physically-correct wall-contact dynamics for
    that step -- u has no effect on theta while pinned (wall reaction absorbs
    it, see cartpole_ode.CartPoleGroundTruthPinned). Without this, a policy
    can exploit an unphysical "free torque through the wall" loophole -- this
    is exactly what our first curriculum-trained policy learned to do.

    track_limit=None disables the wall entirely -- used to test whether the
    policy is genuinely stabilizing via feedback or exploiting the wall's
    "free damping" (velocity gets zeroed on contact) as a crutch.
    """
    batch = x0.shape[0]
    states = torch.zeros(batch, T + 1, 4)
    controls = torch.zeros(batch, T, 1)
    x = apply_track_limits(x0, track_limit) if track_limit is not None else x0
    states[:, 0] = x

    pinned_integrator = None
    if use_wall_physics and track_limit is not None:
        pinned_integrator = make_pinned_integrator(Ts=float(integrator.h))

    for k in range(T):
        u = policy(x)
        controls[:, k] = u
        if pinned_integrator is not None:
            pos = x[:, 0:1]
            at_right = pos >= (track_limit - 1e-6)
            at_left = pos <= (-track_limit + 1e-6)
            pinned_mask = (at_right & (u > 0)) | (at_left & (u < 0))
            x_free = integrator(x, u)
            x_pinned = pinned_integrator(x, u)
            x = torch.where(pinned_mask, x_pinned, x_free)
        else:
            x = integrator(x, u)
        if track_limit is not None:
            x = apply_track_limits(x, track_limit)
        states[:, k + 1] = x
    return states, controls


def evaluate(ckpt_path="policy_balance.pth", T=250, num_test=200,
             success_theta_thresh=0.3, seed=123, use_identified=False,
             dynamics_ckpt="sysid.pth",
             no_wall=True, theta_center=0.0, theta_range=0.3,
             pos_range=0.5, vel_range=0.5, theta_dot_range=0.5,
             policy_hsizes=[60, 60]):
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = CartPolePolicy(nx=4, F_max=F_MAX, hsizes=policy_hsizes)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()

    # same distribution the policy was trained on (override via CLI for e.g. swing-up)
    x0 = sample_balance_x0(num_test, seed=seed, theta_center=theta_center,
                            theta_range=theta_range, pos_range=pos_range,
                            vel_range=vel_range, theta_dot_range=theta_dot_range)

    if use_identified:
        transition = load_frozen_dynamics(dynamics_ckpt)
        states, controls = closed_loop_rollout_identified(policy, transition, x0, T)
        mode_label = "IDENTIFIED model (what the policy was actually trained against)"
    else:
        integrator = make_ground_truth_integrator(Ts=Ts)
        track_limit_used = None if no_wall else TRACK_LIMIT
        states, controls = closed_loop_rollout_ground_truth(policy, integrator, x0, T, track_limit=track_limit_used)
        mode_label = "GROUND-TRUTH physics (NO WALL)" if no_wall else "GROUND-TRUTH physics"

    theta = states[:, :, 2]
    pos = states[:, :, 0]

    # theta is NOT wrapped anywhere in the simulator (raw, unbounded) -- a
    # trajectory that lands at theta=+-2*pi*k (k=1,2,...) is PHYSICALLY at
    # upright too (identical pole orientation), but raw |theta|<thresh would
    # wrongly mark it as failed. wrap into (-pi, pi] before the success check
    # to avoid this. (Especially relevant once loss_theta_state uses a
    # circular loss during training -- that loss doesn't penalize landing at
    # 2*pi vs 0 at all, since they're identical mod 2*pi, so the policy is
    # free to converge to either.)
    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    theta_wrapped = wrap_to_pi(theta)

    # "success" = stays within success_theta_thresh (rad) of upright for the
    # WHOLE rollout after settling (check the last 20% of the horizon, to allow
    # a brief initial transient)
    tail = states[:, int(T * 0.8):, :]
    tail_theta_wrapped = wrap_to_pi(tail[:, :, 2])
    stayed_up_raw = (tail[:, :, 2].abs() < success_theta_thresh).all(dim=1)
    stayed_up_wrapped = (tail_theta_wrapped.abs() < success_theta_thresh).all(dim=1)
    success_rate = stayed_up_raw.float().mean().item()
    success_rate_wrapped = stayed_up_wrapped.float().mean().item()

    print("=" * 60)
    print(f"[{mode_label}: {T} steps = {T*Ts:.2f}s, {num_test} initial conditions]")
    print(f"  Success rate (raw |theta| < {success_theta_thresh} rad, last 20%): "
          f"{success_rate*100:.1f}%")
    print(f"  Success rate (WRAPPED |theta| < {success_theta_thresh} rad, last 20%): "
          f"{success_rate_wrapped*100:.1f}%  <- compare to raw above")
    theta_final_mean = theta[:, -1].abs().mean().item()
    theta_final_max = theta[:, -1].abs().max().item()
    theta_final_wrapped_mean = theta_wrapped[:, -1].abs().mean().item()
    theta_final_wrapped_max = theta_wrapped[:, -1].abs().max().item()
    print(f"  Final |theta| (raw): mean={theta_final_mean:.4f} rad "
          f"({np.degrees(theta_final_mean):.2f} deg), "
          f"max={theta_final_max:.4f} rad ({np.degrees(theta_final_max):.2f} deg)")
    print(f"  Final |theta| (wrapped): mean={theta_final_wrapped_mean:.4f} rad "
          f"({np.degrees(theta_final_wrapped_mean):.2f} deg), "
          f"max={theta_final_wrapped_max:.4f} rad ({np.degrees(theta_final_wrapped_max):.2f} deg)")
    print(f"  Final |cart_pos|: mean={pos[:,-1].abs().mean():.4f} m, max={pos[:,-1].abs().max():.4f} m")
    print(f"  Control effort |u|: mean={controls.abs().mean():.4f} N, max={controls.abs().max():.4f} N "
          f"(F_max={F_MAX} N)")
    hit_wall = (pos.abs() >= TRACK_LIMIT - 1e-3).any(dim=1).float().mean().item()
    print(f"  Fraction reaching |x|>={TRACK_LIMIT}m at some point: {hit_wall*100:.1f}%"
          + (" (no wall -- just a marker, not a clip)" if no_wall else ""))

    # ---------- Plots ----------
    t = np.arange(T + 1) * Ts
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    n_show = min(20, num_test)
    for i in range(n_show):
        axes[0, 0].plot(t, theta[i].numpy(), alpha=0.5, lw=1)
    axes[0, 0].axhline(0, color="black", ls="--", lw=1)
    axes[0, 0].set_title(f"theta over time ({n_show} sample trajectories)")
    axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("theta [rad]")

    for i in range(n_show):
        axes[0, 1].plot(t, pos[i].numpy(), alpha=0.5, lw=1)
    axes[0, 1].axhline(TRACK_LIMIT, color="red", ls="--", lw=1, label="track limit")
    axes[0, 1].axhline(-TRACK_LIMIT, color="red", ls="--", lw=1)
    axes[0, 1].set_title(f"cart_pos over time ({n_show} sample trajectories)")
    axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("x [m]")
    axes[0, 1].legend()

    axes[1, 0].hist(theta[:, -1].numpy(), bins=30)
    axes[1, 0].set_title(f"Final theta distribution (t={T*Ts:.1f}s)")
    axes[1, 0].set_xlabel("theta [rad]")

    tu = np.arange(T) * Ts
    for i in range(n_show):
        axes[1, 1].plot(tu, controls[i, :, 0].numpy(), alpha=0.5, lw=1)
    axes[1, 1].axhline(F_MAX, color="red", ls="--", lw=1)
    axes[1, 1].axhline(-F_MAX, color="red", ls="--", lw=1)
    axes[1, 1].set_title(f"Control input u ({n_show} sample trajectories)")
    axes[1, 1].set_xlabel("time [s]"); axes[1, 1].set_ylabel("u [N]")

    plt.tight_layout()
    if use_identified:
        out_name = "balance_policy_eval_identified.png"
    elif no_wall:
        out_name = "balance_policy_eval_groundtruth_nowall.png"
    else:
        out_name = "balance_policy_eval_groundtruth.png"
    plt.savefig(out_name, dpi=130)
    print(f"\nSaved plot to {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="policy_balance.pth")
    parser.add_argument("--T", type=int, default=250, help="rollout steps (250 = 5s)")
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--use_identified", action="store_true",
                         help="roll out against the identified (training-time) model instead of ground truth")
    parser.add_argument("--dynamics_ckpt", type=str, default="sysid.pth")
    parser.add_argument("--with_wall", action="store_true",
                         help="enable the physical track wall for comparison "
                              "-- default (no flag) is the new wall-less design")
    parser.add_argument("--theta_center", type=float, default=0.0,
                         help="center of theta0 sampling range, e.g. pi (~3.14159) for swing-up tests")
    parser.add_argument("--theta_range", type=float, default=0.3)
    parser.add_argument("--pos_range", type=float, default=0.5)
    parser.add_argument("--vel_range", type=float, default=0.5)
    parser.add_argument("--theta_dot_range", type=float, default=0.5)
    parser.add_argument("--policy_hsizes", type=int, nargs="+", default=[60, 60],
                         help="policy network hidden layer sizes, MUST match the checkpoint")
    args = parser.parse_args()
    evaluate(args.ckpt, T=args.T, num_test=args.num_test,
              use_identified=args.use_identified, dynamics_ckpt=args.dynamics_ckpt,
              no_wall=not args.with_wall, theta_center=args.theta_center, theta_range=args.theta_range,
              pos_range=args.pos_range, vel_range=args.vel_range, theta_dot_range=args.theta_dot_range,
              policy_hsizes=args.policy_hsizes)
