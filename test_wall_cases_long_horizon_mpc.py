"""Test whether the 13 DPC-wall-contact cases (extract_wall_contact_cases.py)
can be solved WITHOUT touching the wall, given a much longer MPC horizon.

If even a long horizon fails to avoid the wall on these specific cases, that's
evidence the wall contact reflects a genuine physical limit (F_max=5N
insufficient to recover in time from these particular states) rather than a
policy/model limitation. If MPC succeeds where DPC failed, it points to DPC
(or the identified model, or short-horizon planning) as the limiting factor.
"""
import time
import numpy as np
import torch

from mpc_baselines import build_fixed_mpc
from dpc_policy_balance import F_MAX
from cartpole_ode import make_ground_truth_integrator
from dpc_policy_balance import Ts, TRACK_LIMIT

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="wall_contact_cases.pt")
    parser.add_argument("--N_mpc", type=int, default=100)
    parser.add_argument("--T", type=int, default=250)
    args = parser.parse_args()

    data = torch.load(args.cases)
    x0_batch = data["x0"]
    orig_indices = data["indices"]
    print(f"'{args.cases}'에서 {len(x0_batch)}개 벽 접촉 케이스 로드 (원본 인덱스: {orig_indices.tolist()})")
    print(f"N_mpc={args.N_mpc}로 재시도 중...")

    integrator = make_ground_truth_integrator(Ts=Ts)
    problem, x0_param, u_var = build_fixed_mpc(N_mpc=args.N_mpc)

    batch = x0_batch.shape[0]
    states = torch.zeros(batch, args.T + 1, 4)
    fail_count = 0
    fail_count_per_traj = [0] * batch
    status_counts = {}
    first_u_per_traj = [None] * batch
    t0 = time.time()
    for i in range(batch):
        x = x0_batch[i:i + 1]
        states[i, 0] = x[0]
        for k in range(args.T):
            x0_param.value = x[0].numpy()
            problem.solve(solver="OSQP", warm_start=True,
                          max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
            status_counts[problem.status] = status_counts.get(problem.status, 0) + 1
            if u_var.value is None:
                fail_count += 1
                fail_count_per_traj[i] += 1
                u_applied = 0.0
            else:
                u_applied = float(np.clip(u_var.value[0, 0], -F_MAX, F_MAX))
            if k == 0:
                first_u_per_traj[i] = u_applied
            u = torch.tensor([[u_applied]], dtype=torch.float32)
            x = integrator(x, u)
            states[i, k + 1] = x[0]
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (batch - i - 1)
        print(f"  ...{i+1}/{batch} 완료 (원본 idx={orig_indices[i].item()}), 경과 {elapsed:.0f}s, 예상 잔여 {eta:.0f}s")
    if fail_count > 0:
        print(f"  (경고: QP solve 실패 {fail_count}/{batch*args.T} 스텝)")
    print(f"  Solver status 분포: {status_counts}")

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
    print(f"[벽 접촉 케이스 {batch}개, MPC N_mpc={args.N_mpc}로 재시도]")
    for i in range(batch):
        print(f"  idx={orig_indices[i].item():3d}: 벽접촉={'예' if hit_wall[i] else '아니오'}, "
              f"성공(raw)={'예' if success_raw[i] else '아니오'}, "
              f"성공(wrapped)={'예' if success_wrapped[i] else '아니오'}, "
              f"solve실패={fail_count_per_traj[i]}/{args.T}, "
              f"첫스텝u={first_u_per_traj[i]:.3f}N, "
              f"final|theta|(raw)={theta[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|theta|(wrapped)={theta_wrapped[i,-1].abs().item()*180/np.pi:.1f}deg, "
              f"final|pos|={pos[i,-1].abs().item():.3f}m, "
              f"max|pos|={pos[i].abs().max().item():.3f}m")
    print(f"\n  벽 접촉 여전히 발생: {hit_wall.sum().item()}/{batch}")
    print(f"  성공(raw): {success_raw.sum().item()}/{batch}, 성공(wrapped): {success_wrapped.sum().item()}/{batch}")
    print(f"\n  해석: 벽 접촉이 {'0에 가까우면' if hit_wall.sum().item() <= 2 else '여전히 많으면'} "
          f"{'DPC/짧은 horizon의 한계였다는 뜻' if hit_wall.sum().item() <= 2 else 'F_max=5N 물리적 한계에 가깝다는 뜻'}")
