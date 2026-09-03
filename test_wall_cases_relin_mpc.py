"""Test the 13 saved wall-contact cases (wall_contact_cases.pt) with the
RE-LINEARIZED MPC (re-linearizes the nonlinear dynamics at the current state
every step, see compare_relinearized_mpc.py) instead of the fixed
(origin-linearized) MPC. If fixed-linearization MPC failed because the model
is too inaccurate this far from the origin, re-linearization should fix it
(or at least clearly improve it); if it STILL fails just as badly, the
fixed-linearization hypothesis is wrong and something else is going on.
"""
import time
import numpy as np
import torch

from mpc_baselines import solve_relin_mpc_step
from cartpole_ode import make_ground_truth_integrator
from dpc_policy_balance import Ts, TRACK_LIMIT

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="wall_contact_cases.pt")
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--N_mpc", type=int, default=15)
    args = parser.parse_args()

    data = torch.load(args.cases)
    x0_batch = data["x0"]
    orig_indices = data["indices"]
    print(f"'{args.cases}'에서 {len(x0_batch)}개 벽 접촉 케이스 로드 (원본 인덱스: {orig_indices.tolist()})")
    print(f"재선형화 MPC (N_mpc={args.N_mpc})로 재시도 중 (느립니다, 기다려주세요)...")

    integrator = make_ground_truth_integrator(Ts=Ts)
    batch = x0_batch.shape[0]
    states = torch.zeros(batch, args.T + 1, 4)
    t0 = time.time()
    for i in range(batch):
        x = x0_batch[i:i + 1]
        states[i, 0] = x[0]
        for k in range(args.T):
            u_applied = solve_relin_mpc_step(x[0].numpy(), N_mpc=args.N_mpc)
            u = torch.tensor([[u_applied]], dtype=torch.float32)
            x = integrator(x, u)
            states[i, k + 1] = x[0]
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (batch - i - 1)
        print(f"  ...{i+1}/{batch} 완료 (원본 idx={orig_indices[i].item()}), 경과 {elapsed:.0f}s, 예상 잔여 {eta:.0f}s")

    pos = states[:, :, 0]
    theta = states[:, :, 2]

    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    theta_wrapped = wrap_to_pi(theta)
    hit_wall = (pos.abs() >= TRACK_LIMIT - 1e-3).any(dim=1)
    tail = states[:, int(args.T * 0.8):, :]
    tail_theta_wrapped = wrap_to_pi(tail[:, :, 2])
    success_raw = (tail[:, :, 2].abs() < 0.3).all(dim=1)
    success_wrapped = (tail_theta_wrapped.abs() < 0.3).all(dim=1)

    print("\n" + "=" * 60)
    print(f"[벽 접촉 케이스 {batch}개, 재선형화 MPC로 재시도]")
    for i in range(batch):
        print(f"  idx={orig_indices[i].item():3d}: 벽접촉={'예' if hit_wall[i] else '아니오'}, "
              f"성공(raw)={'예' if success_raw[i] else '아니오'}, "
              f"성공(wrapped)={'예' if success_wrapped[i] else '아니오'}, "
              f"final|theta|(raw)={theta[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|theta|(wrapped)={theta_wrapped[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|pos|={pos[i,-1].abs().item():.3f}m, "
              f"max|pos|={pos[i].abs().max().item():.3f}m")
    print(f"\n  벽 접촉 여전히 발생: {hit_wall.sum().item()}/{batch}")
    print(f"  성공(raw): {success_raw.sum().item()}/{batch}, 성공(wrapped): {success_wrapped.sum().item()}/{batch}")
