"""Curriculum training for the cart-pole balancing DPC policy.

Two fixes applied based on check_policy_sign.py diagnosis:
1. theta_dot sign was learned backwards, policy(0)=2.99N instead of ~0 --
   chicken-and-egg problem: the policy never stabilized well enough during
   training to actually visit near-equilibrium states, so it never learned
   what to do there. Fix: start training with a VERY narrow initial-condition
   range (near equilibrium), then progressively widen it -- similar in spirit
   to the sysID N_pred curriculum.
2. Training hadn't converged (still improving at epoch 99/100). Fix: much
   longer training budget per stage (Euler integration is cheap).

Policy network (CartPolePolicy) doesn't depend on the IC range, only on
nx/F_max, so warm-starting across stages is direct -- same pattern as the
sysID curriculum (train_nssm_curriculum.py).
"""
import torch
from dpc_policy_balance import (
    CartPolePolicy, build_balance_problem, make_balance_dataset,
    load_frozen_dynamics, PrintDevLossCallback, F_MAX,
)
from neuromancer.trainer import Trainer

N = 80  # rollout horizon, kept fixed across stages (within trusted dynamics region)
DYNAMICS_CKPT = "sysid.pth"
HSIZES = [60, 60]  # capacity sweep 결과 128x3이 더 낫지 않음을 확인함 -> 원래(60,60)로 복귀

# (theta_range, pos_range, vel_range, theta_dot_range, epochs, patience)
STAGES = [
    (0.05, 0.1, 0.1, 0.1, 150, 25),
    (0.15, 0.3, 0.3, 0.3, 150, 25),
    (0.30, 0.5, 0.5, 0.5, 200, 30),
]

if __name__ == "__main__":
    print("Frozen dynamics 로드 중...")
    transition = load_frozen_dynamics(DYNAMICS_CKPT)

    policy = CartPolePolicy(nx=4, F_max=F_MAX, hsizes=HSIZES)
    for stage_i, (theta_r, pos_r, vel_r, theta_dot_r, epochs, patience) in enumerate(STAGES):
        print("=" * 70)
        print(f"[Stage {stage_i}] theta_range=+-{theta_r}, pos_range=+-{pos_r}, "
              f"vel_range=+-{vel_r}, theta_dot_range=+-{theta_dot_r}, epochs={epochs}, patience={patience}")
        print("=" * 70)

        x0_kwargs = dict(pos_range=pos_r, vel_range=vel_r, theta_range=theta_r, theta_dot_range=theta_dot_r)
        train_dataset, train_loader = make_balance_dataset(2000, N, seed=stage_i, name="train", **x0_kwargs)
        dev_dataset, dev_loader = make_balance_dataset(400, N, seed=1000 + stage_i, name="dev", **x0_kwargs)

        problem, policy = build_balance_problem(transition, N=N, policy=policy)
        if stage_i > 0:
            print(f"  -> warm-started from stage {stage_i - 1}")

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        trainer = Trainer(
            problem, train_loader, dev_loader, optimizer=optimizer,
            epochs=epochs, patience=patience, warmup=5,
            train_metric="train_loss", dev_metric="dev_loss", eval_metric="dev_loss",
            callback=PrintDevLossCallback(),
        )
        best_model = trainer.train()
        problem.load_state_dict(best_model)

        ckpt_name = f"policy_balance_stage{stage_i}.pth"
        torch.save(policy.state_dict(), ckpt_name)
        print(f"  Saved: {ckpt_name}  (best dev loss: {trainer.best_devloss.item():.4f})")

        # quick equilibrium sanity check after each stage
        with torch.no_grad():
            u0 = policy(torch.zeros(1, 4)).item()
        print(f"  policy(0,0,0,0) = {u0:.4f}  (want close to 0)")

    torch.save(policy.state_dict(), "policy_balance.pth")
    print("\nCurriculum training complete. Final policy: policy_balance.pth")
