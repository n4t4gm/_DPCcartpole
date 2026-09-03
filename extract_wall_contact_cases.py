"""Extract the specific initial conditions (from the standard test set:
seed=123, num_test=200, theta0+-0.3/pos,vel,theta_dot0+-0.5, same as
eval_balance_policy.py) where the DPC policy's cart reaches |x|>=1.5m at
some point during the rollout.

Saves them to a .pt file for reuse -- e.g. testing whether a much longer-
horizon MPC can avoid the wall on these SAME cases, to determine whether
wall contact reflects a genuine physical limit (F_max=5N insufficient) or
just a DPC policy limitation.
"""
import torch

from cartpole_ode import make_ground_truth_integrator
from dpc_policy_balance import CartPolePolicy, sample_balance_x0, F_MAX, TRACK_LIMIT, Ts
from eval_balance_policy import closed_loop_rollout_ground_truth

CKPT = "policy_balance.pth"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="wall_contact_cases.pt")
    args = parser.parse_args()

    policy = CartPolePolicy(nx=4, F_max=F_MAX)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()

    torch.manual_seed(args.seed)
    x0_batch = sample_balance_x0(args.num_test, seed=args.seed)  # matches eval_balance_policy.py defaults

    integrator = make_ground_truth_integrator(Ts=Ts)
    states, controls = closed_loop_rollout_ground_truth(policy, integrator, x0_batch, args.T, track_limit=None)

    pos = states[:, :, 0]
    hit_wall = (pos.abs() >= TRACK_LIMIT - 1e-3).any(dim=1)
    idx = hit_wall.nonzero(as_tuple=True)[0]

    print(f"전체 {args.num_test}개 중 벽 접촉 케이스: {len(idx)}개")
    print(f"인덱스: {idx.tolist()}")

    x0_wall = x0_batch[idx]
    torch.save({
        "x0": x0_wall,
        "indices": idx,
        "seed": args.seed,
        "num_test": args.num_test,
        "T": args.T,
        "ckpt": args.ckpt,
    }, args.out)
    print(f"\n저장 완료: {args.out} ({len(idx)}개 초기조건)")
    print("x0 값들 (pos, vel, theta, theta_dot):")
    for i, x0 in zip(idx.tolist(), x0_wall):
        print(f"  idx={i}: {x0.numpy().round(3)}")
