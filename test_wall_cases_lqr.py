"""Test the 13 saved wall-contact cases with LQR (infinite-horizon, computed
via scipy.linalg.solve_continuous_are around the upright equilibrium -- same
gain used in check_policy_sign.py, previously verified to give 76.7% on the
standard 30-case comparison test).

Specifically want to see what happens on idx 48, 74, 188 -- the 3 cases where
DPC (wall-less, verified) itself failed (theta didn't converge). If LQR ALSO
fails on these 3, that's evidence they're genuinely hard (though LQR being
linear-around-origin is itself a weak controller for large deviations, so
failure here doesn't strongly prove physical impossibility). If LQR succeeds
where DPC failed, that's strong evidence DPC specifically mislearned these
points rather than them being infeasible.
"""
import numpy as np
import torch

from cartpole_ode import make_ground_truth_integrator
from dpc_policy_balance import Ts, F_MAX, TRACK_LIMIT

K = torch.tensor([[-10.0, -12.23870008, 78.07563937, 18.67741601]])  # u_lqr = -K @ x


def lqr_u(x):
    u = -(x @ K.T)
    return torch.clamp(u, -F_MAX, F_MAX)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="wall_contact_cases.pt")
    parser.add_argument("--T", type=int, default=250)
    args = parser.parse_args()

    data = torch.load(args.cases)
    x0_batch = data["x0"]
    orig_indices = data["indices"]
    print(f"'{args.cases}'에서 {len(x0_batch)}개 케이스 로드 (원본 인덱스: {orig_indices.tolist()})")
    print("LQR로 재시도 중 (빠름)...")

    integrator = make_ground_truth_integrator(Ts=Ts)
    batch = x0_batch.shape[0]
    T = args.T
    states = torch.zeros(batch, T + 1, 4)
    x = x0_batch.clone()
    states[:, 0] = x
    with torch.no_grad():
        for k in range(T):
            u = lqr_u(x)
            x = integrator(x, u)
            states[:, k + 1] = x

    pos = states[:, :, 0]
    theta = states[:, :, 2]

    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    theta_wrapped = wrap_to_pi(theta)
    hit_wall = (pos.abs() >= TRACK_LIMIT - 1e-3).any(dim=1)
    tail = states[:, int(T * 0.8):, :]
    tail_theta_wrapped = wrap_to_pi(tail[:, :, 2])
    success_wrapped = (tail_theta_wrapped.abs() < 0.3).all(dim=1)

    print("\n" + "=" * 60)
    print(f"[벽 접촉 케이스 {batch}개, LQR로 재시도]")
    for i in range(batch):
        print(f"  idx={orig_indices[i].item():3d}: 성공(wrapped)={'예' if success_wrapped[i] else '아니오'}, "
              f"벽근처={'예' if hit_wall[i] else '아니오'}, "
              f"final|theta|(wrapped)={theta_wrapped[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|pos|={pos[i,-1].abs().item():.3f}m, max|pos|={pos[i].abs().max().item():.3f}m")
    print(f"\n  성공(wrapped): {success_wrapped.sum().item()}/{batch}")
