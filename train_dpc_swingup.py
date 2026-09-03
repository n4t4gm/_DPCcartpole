"""Swing-up DPC policy: warm-started from the balancing policy
(policy_balance.pth), but now sampling theta0
over the FULL range [-pi, pi] in one shot -- matching Amos et al. 2018,
ICLR DiffMPC 2026, and KCPO 2023, all of which train cartpole swing-up this
way (never narrowing to "near pi only" first). This also sidesteps the
catastrophic forgetting seen in the earlier narrow-range attempt: theta~0
states are always present in the training distribution too, so balancing
behavior can't be forgotten the way it was when training was 100% theta~=pi.

Same loss structure as balancing (quadratic tracking to x=0, i.e. upright +
centered). No circular/wrapped theta loss added here -- unlike sysID (many
full rotations from PRBS), a single swing-up trajectory only needs to cover
roughly [0, pi] once, so the raw quadratic-to-0 loss should behave reasonably
without needing to handle the +-pi branch cut.
"""
import torch
import numpy as np
from dpc_policy_balance import (
    CartPolePolicy, build_balance_problem, make_balance_dataset,
    load_frozen_dynamics, PrintDevLossCallback, F_MAX,
)
from neuromancer.trainer import Trainer

N = 150  # 3.0s -- pendulum period ~1.6s, need room for one swing + settle
DYNAMICS_CKPT = "sysid.pth"
WARM_START_CKPT = "policy_balance.pth"

if __name__ == "__main__":
    print("Frozen dynamics 로드 중...")
    transition = load_frozen_dynamics(DYNAMICS_CKPT)

    print(f"Balancing policy로 warm-start: {WARM_START_CKPT}")
    policy = CartPolePolicy(nx=4, F_max=F_MAX)
    policy.load_state_dict(torch.load(WARM_START_CKPT, map_location="cpu"))

    # theta0 over the FULL range, matching Amos et al./ICLR DiffMPC/KCPO
    # (pos, vel, theta_dot ranges also matched to their setup)
    x0_kwargs = dict(pos_range=0.5, vel_range=0.5, theta_range=np.pi,
                      theta_dot_range=1.0, theta_center=0.0)

    train_dataset, train_loader = make_balance_dataset(2000, N, seed=0, name="train", **x0_kwargs)
    dev_dataset, dev_loader = make_balance_dataset(400, N, seed=1, name="dev", **x0_kwargs)

    # R (control effort penalty) 대폭 상향 -- balancing 기본값(0.01)으로는
    # swing-up 궤적에서 u가 5초 내내 +-F_max 사이를 chattering하는 게
    # 관찰됨(balance_policy_eval_groundtruth_nowall.png), 에너지 펌핑 뒤
    # 부드럽게 감속/안정화하는 대신 계속 최대출력으로 떠는 전략에 갇힘.
    # balancing 쪽 기본값(R=0.01)은 안 건드림 (거기선 잘 작동 중, 98.5%).
    # R을 0.3에서 0.1로 완화 -- 0.3은 chattering은 잡았지만 policy가
    # theta=pi(에너지가 더 낮은 안정 평형점)에 안주하게 만든 부작용이 있었음.
    # energy_weight를 추가해서 "pi에 가만히 있는 것"에 명시적으로 벌점을 줌
    # (그 지점은 upright보다 에너지가 낮아서, 거리 기반 loss_state만으로는
    # 최적화가 그리로 안주하는 걸 못 막았음 -- 대화 참고).
    problem, policy = build_balance_problem(transition, N=N, policy=policy, R=0.1, energy_weight=10.0)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    trainer = Trainer(
        problem, train_loader, dev_loader, optimizer=optimizer,
        epochs=400, patience=40, warmup=5,  # was 200/30 -- 지난 학습이 epoch 199까지도
        # badcount<patience로 계속 개선 중이었음 (21.08 loss, 여전히 하강 추세)
        train_metric="train_loss", dev_metric="dev_loss", eval_metric="dev_loss",
        callback=PrintDevLossCallback(),
    )

    print("Swing-up 학습 시작!")
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    torch.save(policy.state_dict(), "policy_swingup.pth")
    print("\n학습 완료! 'policy_swingup.pth' 저장됨.")

    with torch.no_grad():
        u0 = policy(torch.zeros(1, 4)).item()
    print(f"policy(0,0,0,0) = {u0:.4f}  (upright/centered -> want close to 0)")
