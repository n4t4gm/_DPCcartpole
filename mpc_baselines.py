"""Unified linear-MPC (QP, OSQP) baselines vs DPC comparison. Consolidates
what used to be 4 separate scripts:
  - compare_mpc_vs_dpc.py       (fixed-linearization MPC vs DPC)
  - compare_relinearized_mpc.py (re-linearized/successive-linearization MPC vs DPC)
  - eval_mpc_wall_contact.py    (fixed-linearization MPC on the standard
                                  eval_balance_policy.py test distribution)
  - eval_relin_mpc_wall_contact.py (same, for re-linearized MPC)

Two MPC variants are provided:
  - FIXED linearization: linearize ONCE at the upright equilibrium (x=0),
    reuse that (A,B) for the whole rollout. Cheap, but inaccurate far from
    the origin.
  - RE-LINEARIZED (successive linearization): re-linearize the nonlinear
    dynamics at the CURRENT state every control step. More locally accurate,
    but much slower (fresh Jacobian + QP build every step) -- and, as found
    in this project, this extra accuracy does NOT reliably translate into
    better closed-loop performance (see conversation: it tends to get stuck
    near theta=pi, a stable equilibrium of the free pendulum, when the
    horizon is short).

Both use the SAME linearization point machinery (numerical Jacobian of the
ground-truth ODE, via cartpole_ode.CartPoleGroundTruth -- no dependency on
the old numpy dynamics.py module, which has been removed from this repo).

Terminal cost is the discrete-time infinite-horizon cost-to-go (via
scipy.linalg.solve_discrete_are), NOT raw Q -- using raw Q was found to
catastrophically underweight the cost beyond the horizon (see conversation).

Usage:
    python mpc_baselines.py --mode compare --linearization fixed
    python mpc_baselines.py --mode compare --linearization relin
    python mpc_baselines.py --mode wall_test --linearization fixed --N_mpc 60
    python mpc_baselines.py --mode wall_test --linearization relin
"""
import time
import numpy as np
import torch
import cvxpy as cp
from scipy.linalg import expm, solve_discrete_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_ode import make_ground_truth_integrator, CartPoleGroundTruth
from dpc_policy_balance import CartPolePolicy, sample_balance_x0, F_MAX, Ts, TRACK_LIMIT

nx, nu = 4, 1
Q = np.diag([8.0, 0.1, 10.0, 0.1])  # same weights as the DPC balance policy loss
R = np.array([[0.01]])
N_MPC_DEFAULT = 15

_ODE = CartPoleGroundTruth()  # for computing the nonlinear vector field at any state


def _nl_dynamics(x, u):
    """Continuous-time nonlinear dynamics dx/dt = f(x,u), evaluated via the
    same ground-truth ODE used for simulation (cartpole_ode.py) -- avoids
    needing the old standalone numpy dynamics.py module.
    x, u: (4,), scalar -> returns (4,) numpy array.
    """
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        ut = torch.tensor([[u]], dtype=torch.float32)
        dxdt = _ODE.ode_equations(xt, ut)
    return dxdt.squeeze(0).numpy()


# ---------- Fixed linearization (once, at the origin) ----------
def _linearize_at_origin():
    x0 = np.zeros(4)
    eps = 1e-5
    f0 = _nl_dynamics(x0, 0.0)
    A_c = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        A_c[:, i] = (_nl_dynamics(x0 + dx, 0.0) - f0) / eps
    B_c = ((_nl_dynamics(x0, eps) - f0) / eps).reshape(4, 1)
    return A_c, B_c


def _discretize_linear(A_c, B_c, Ts):
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A_c
    M[:nx, nx:] = B_c
    Md = expm(M * Ts)
    return Md[:nx, :nx], Md[:nx, nx:]


_A_c_fixed, _B_c_fixed = _linearize_at_origin()
A_D_FIXED, B_D_FIXED = _discretize_linear(_A_c_fixed, _B_c_fixed, Ts)
# Terminal cost MUST be the infinite-horizon cost-to-go (discrete ARE
# solution), not raw Q -- see module docstring / conversation.
P_TERMINAL_FIXED = solve_discrete_are(A_D_FIXED, B_D_FIXED, Q, R)


def build_fixed_mpc(N_mpc=N_MPC_DEFAULT):
    """Parametric QP (parameter: x0) using the fixed origin-linearization;
    built once, re-solved (with warm-start) every control step."""
    x0_param = cp.Parameter(nx)
    x_var = cp.Variable((nx, N_mpc + 1))
    u_var = cp.Variable((nu, N_mpc))

    cost = 0
    constraints = [x_var[:, 0] == x0_param]
    for k in range(N_mpc):
        cost += cp.quad_form(x_var[:, k], Q) + cp.quad_form(u_var[:, k], R)
        constraints += [x_var[:, k + 1] == A_D_FIXED @ x_var[:, k] + B_D_FIXED @ u_var[:, k]]
        constraints += [cp.abs(u_var[:, k]) <= F_MAX]
    cost += cp.quad_form(x_var[:, N_mpc], P_TERMINAL_FIXED)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, x0_param, u_var


# ---------- Re-linearized (successive linearization, every step) ----------
def _linearize_at(x_bar, eps=1e-5):
    """Numerical Jacobian of the nonlinear dynamics at (x_bar, u=0).
    Away from an equilibrium, f(x_bar,0) != 0, so the affine offset c_c
    matters: xdot ~= A_c x + B_c u + c_c, c_c = f(x_bar,0) - A_c @ x_bar.
    """
    f0 = _nl_dynamics(x_bar, 0.0)
    A_c = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        A_c[:, i] = (_nl_dynamics(x_bar + dx, 0.0) - f0) / eps
    B_c = ((_nl_dynamics(x_bar, eps) - f0) / eps).reshape(4, 1)
    c_c = f0 - A_c @ x_bar
    return A_c, B_c, c_c


def _discretize_affine(A_c, B_c, c_c, Ts):
    """Exact (zero-order hold) discretization of xdot = A_c x + B_c u + c_c,
    via the augmented-state matrix exponential trick."""
    M = np.zeros((nx + 1 + nu, nx + 1 + nu))
    M[:nx, :nx] = A_c
    M[:nx, nx:nx + 1] = c_c.reshape(-1, 1)
    M[:nx, nx + 1:] = B_c
    Md = expm(M * Ts)
    A_d = Md[:nx, :nx]
    c_d = Md[:nx, nx:nx + 1].flatten()
    B_d = Md[:nx, nx + 1:]
    return A_d, B_d, c_d


def solve_relin_mpc_step(x_bar, N_mpc=N_MPC_DEFAULT):
    """Re-linearize at x_bar, build a fresh QP over the horizon, solve,
    return the first control action. Much slower than the fixed-linearization
    path (new Jacobian + QP build every call)."""
    A_d, B_d, c_d = _discretize_affine(*_linearize_at(x_bar), Ts)
    try:
        P_terminal = solve_discrete_are(A_d, B_d, Q, R)
    except Exception:
        P_terminal = Q  # fallback if ARE fails to converge at this point (rare)

    x0_p = cp.Parameter(nx)
    x0_p.value = x_bar
    x_var = cp.Variable((nx, N_mpc + 1))
    u_var = cp.Variable((nu, N_mpc))

    cost = 0
    constraints = [x_var[:, 0] == x0_p]
    for k in range(N_mpc):
        cost += cp.quad_form(x_var[:, k], Q) + cp.quad_form(u_var[:, k], R)
        constraints += [x_var[:, k + 1] == A_d @ x_var[:, k] + B_d @ u_var[:, k] + c_d]
        constraints += [cp.abs(u_var[:, k]) <= F_MAX]
    cost += cp.quad_form(x_var[:, N_mpc], P_terminal)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, warm_start=True)
    if u_var.value is None:
        return 0.0
    return float(np.clip(u_var.value[0, 0], -F_MAX, F_MAX))


# ---------- Rollouts ----------
@torch.no_grad()
def run_dpc(policy, x0_batch, T):
    """Closed-loop DPC rollout on ground truth, timing each forward pass."""
    integrator = make_ground_truth_integrator(Ts=Ts)
    batch = x0_batch.shape[0]
    for _ in range(5):  # warm-up (exclude one-time init overhead from timing)
        _ = policy(x0_batch)
    states = torch.zeros(batch, T + 1, 4)
    states[:, 0] = x0_batch
    x = x0_batch
    times = []
    for k in range(T):
        t0 = time.perf_counter()
        u = policy(x)
        times.append(time.perf_counter() - t0)
        x = integrator(x, u)
        states[:, k + 1] = x
    return states, np.array(times)


def run_fixed_mpc(x0_batch, T, N_mpc=N_MPC_DEFAULT, verbose_every=None):
    """Closed-loop rollout with the fixed-linearization MPC (one trajectory
    at a time -- QP solvers aren't batched the way NN inference is)."""
    integrator = make_ground_truth_integrator(Ts=Ts)
    problem, x0_param, u_var = build_fixed_mpc(N_mpc=N_mpc)
    x0_param.value = x0_batch[0].numpy()
    for _ in range(3):  # warm-up (exclude cvxpy's one-time compilation)
        problem.solve(solver=cp.OSQP, warm_start=True)

    batch = x0_batch.shape[0]
    states = torch.zeros(batch, T + 1, 4)
    times, fail_count = [], 0
    t_start = time.time()
    for i in range(batch):
        x = x0_batch[i:i + 1]
        states[i, 0] = x[0]
        for k in range(T):
            x0_param.value = x[0].numpy()
            t0 = time.perf_counter()
            problem.solve(solver=cp.OSQP, warm_start=True)
            times.append(time.perf_counter() - t0)
            if u_var.value is None:
                fail_count += 1
                u_applied = 0.0
            else:
                u_applied = float(np.clip(u_var.value[0, 0], -F_MAX, F_MAX))
            u = torch.tensor([[u_applied]], dtype=torch.float32)
            x = integrator(x, u)
            states[i, k + 1] = x[0]
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (batch - i - 1)
            print(f"  ...{i+1}/{batch} 완료 (경과 {elapsed:.0f}s, 예상 잔여 {eta:.0f}s)")
    if fail_count > 0:
        print(f"  (경고: QP solve 실패 {fail_count}/{batch*T} 스텝)")
    return states, np.array(times)


def run_relin_mpc(x0_batch, T, N_mpc=N_MPC_DEFAULT, verbose_every=None):
    """Closed-loop rollout with the re-linearized MPC."""
    integrator = make_ground_truth_integrator(Ts=Ts)
    batch = x0_batch.shape[0]
    states = torch.zeros(batch, T + 1, 4)
    times = []
    t_start = time.time()
    for i in range(batch):
        x = x0_batch[i:i + 1]
        states[i, 0] = x[0]
        for k in range(T):
            x_np = x[0].numpy()
            t0 = time.perf_counter()
            u_applied = solve_relin_mpc_step(x_np, N_mpc=N_mpc)
            times.append(time.perf_counter() - t0)
            u = torch.tensor([[u_applied]], dtype=torch.float32)
            x = integrator(x, u)
            states[i, k + 1] = x[0]
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (batch - i - 1)
            print(f"  ...{i+1}/{batch} 완료 (경과 {elapsed:.0f}s, 예상 잔여 {eta:.0f}s)")
    return states, np.array(times)


def success_rate(states, T, thresh=0.3):
    tail = states[:, int(T * 0.8):, :]
    return (tail[:, :, 2].abs() < thresh).all(dim=1).float().mean().item()


def wall_contact_rate(states):
    return (states[:, :, 0].abs() >= TRACK_LIMIT - 1e-3).any(dim=1).float().mean().item()


# ---------- CLI modes ----------
def mode_compare(args):
    """DPC vs one MPC variant: solve-time histograms + success rate, on a
    wide (theta/pos/vel/theta_dot-range-configurable) test distribution."""
    print(f"{args.num_test}개 초기조건, {args.T}스텝({args.T*Ts:.1f}s), "
          f"linearization={args.linearization} 비교 중...")
    x0_batch = sample_balance_x0(args.num_test, seed=42, theta_range=args.theta_range,
                                  pos_range=args.pos_range, vel_range=args.vel_range,
                                  theta_dot_range=args.theta_dot_range)

    print("[1/2] DPC policy 롤아웃...")
    policy = CartPolePolicy(nx=4, F_max=F_MAX)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()
    dpc_states, dpc_times = run_dpc(policy, x0_batch, args.T)

    print(f"[2/2] QP-MPC ({args.linearization}) 롤아웃 (한 궤적씩, 시간 좀 걸립니다)...")
    if args.linearization == "fixed":
        mpc_states, mpc_times = run_fixed_mpc(x0_batch, args.T, N_mpc=args.N_mpc, verbose_every=10)
    else:
        mpc_states, mpc_times = run_relin_mpc(x0_batch, args.T, N_mpc=args.N_mpc, verbose_every=5)

    print("\n" + "=" * 60)
    print("[Solve / inference time per control step]")
    print(f"  DPC (MLP forward):  median={np.median(dpc_times)*1e6:8.2f} us")
    print(f"  QP-MPC ({args.linearization}): median={np.median(mpc_times)*1e3:8.3f} ms")
    print(f"  DPC is ~{mpc_times.mean()/dpc_times.mean():,.0f}x faster per control step")

    print(f"\n[Closed-loop success rate, ground truth, +-0.3 rad thresh]")
    print(f"  DPC success rate:    {success_rate(dpc_states, args.T)*100:.1f}%")
    print(f"  QP-MPC success rate: {success_rate(mpc_states, args.T)*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(dpc_times * 1e6, bins=30, alpha=0.7, color="tab:blue")
    axes[0].set_xlabel("time [microseconds]"); axes[0].set_title("DPC inference time per step")
    axes[1].hist(mpc_times * 1e3, bins=30, alpha=0.7, color="tab:orange")
    axes[1].set_xlabel("time [milliseconds]"); axes[1].set_title(f"QP-MPC ({args.linearization}) solve time per step")
    plt.tight_layout()
    out_name = f"mpc_{args.linearization}_vs_dpc_solvetime.png"
    plt.savefig(out_name, dpi=130)
    print(f"\nSaved plot to {out_name}")


def mode_wall_test(args):
    """MPC on the SAME test distribution as eval_balance_policy.py (theta0
    +-0.3, pos/vel/theta_dot0 +-0.5, seed=123) -- directly comparable to the
    DPC policy's reported numbers on that same distribution."""
    print(f"MPC ({args.linearization}) 평가: {args.num_test}개 궤적, {args.T}스텝, N_mpc={args.N_mpc} "
          f"(eval_balance_policy.py와 동일 IC 분포, seed={args.seed})")
    x0_batch = sample_balance_x0(args.num_test, seed=args.seed)

    if args.linearization == "fixed":
        states, _ = run_fixed_mpc(x0_batch, args.T, N_mpc=args.N_mpc, verbose_every=20)
    else:
        states, _ = run_relin_mpc(x0_batch, args.T, N_mpc=args.N_mpc, verbose_every=5)

    print("\n" + "=" * 60)
    print(f"[QP-MPC ({args.linearization}, N_mpc={args.N_mpc}), same test distribution as eval_balance_policy.py]")
    print(f"  Success rate (|theta|<0.3 rad, last 20%): {success_rate(states, args.T)*100:.1f}%")
    print(f"  Fraction reaching |x|>=1.5m: {wall_contact_rate(states)*100:.1f}%")
    print(f"  Final |cart_pos| mean: {states[:,-1,0].abs().mean().item():.4f} m, "
          f"max: {states[:,-1,0].abs().max().item():.4f} m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "wall_test"], default="compare",
                         help="'compare': DPC vs MPC solve-time+success on a configurable IC distribution. "
                              "'wall_test': MPC alone on the standard eval_balance_policy.py test set.")
    parser.add_argument("--linearization", choices=["fixed", "relin"], default="fixed",
                         help="'fixed': linearize once at the origin. 'relin': re-linearize every step (slow).")
    parser.add_argument("--N_mpc", type=int, default=N_MPC_DEFAULT)
    parser.add_argument("--T", type=int, default=150)
    parser.add_argument("--num_test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123, help="only used in wall_test mode")
    parser.add_argument("--theta_range", type=float, default=0.3)
    parser.add_argument("--pos_range", type=float, default=0.5)
    parser.add_argument("--vel_range", type=float, default=0.5)
    parser.add_argument("--theta_dot_range", type=float, default=0.5)
    parser.add_argument("--ckpt", type=str, default="policy_balance.pth")
    args = parser.parse_args()

    if args.mode == "compare":
        mode_compare(args)
    else:
        mode_wall_test(args)
