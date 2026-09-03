"""Fully-observable System Identification.

Unlike train_nssm_sysid.py (which assumes PARTIAL observation: an observer
estimates x0 from a window of past [cart_pos, theta] + u measurements), this
trains the NSSM transition model with the TRUE x0 given directly -- no
observer at all. This isolates a clean question: given the exact initial
state, how accurate is the pure DYNAMICS model (transition) alone?

Motivation: sysID errors seen with the observer-based pipeline (theta RMSE
~15-40 deg depending on horizon) seemed too large for a problem this
well-conditioned (smooth, low-order ODE, informative sin/cos embedding).
This tests whether the observer's estimation error was the dominant
contributor -- if fully-observable sysID gets errors 10-100x smaller, that
confirms it; if errors stay similarly large, the transition model itself
(or the loss/training setup) is the real bottleneck, not partial
observability.

Downstream implication: if this works well, the whole project moves to a
fully-observable assumption throughout (sysID AND policy), matching what
policy training already silently assumed -- removing the current mismatch
where sysID pretends partial observability but policy does not.
"""
import numpy as np
import torch
import torch.nn as nn

from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.callbacks import Callback

from generate_dataset import generate_cartpole_dataset
from nssm_model import (
    CartPoleNSSMTransition, CartPoleNSSMTransitionRK4,
    circular_theta_objective, PrintDevLossCallback,
)


def build_fullobs_sysid_problem(
    N_pred: int = 100, Ts: float = 0.02,
    Qdx: float = 0.05, Qy: float = 1.0,
    y_bounds: tuple = (3.0, 6 * np.pi),
    transition_hsizes: list = [32, 32],
    integration: str = "euler",
) -> Problem:
    """Same loss structure as build_nssm_sysid_problem, MINUS the observer
    entirely -- x_target[:,0,:] (the TRUE x0) is fed directly as the rollout's
    starting state, no estimation involved. Only the transition (NSSM) model
    is being trained/evaluated here.
    """
    if integration == "rk4":
        nssm_transition = CartPoleNSSMTransitionRK4(nx=4, nu=1, dt=Ts, hsizes=transition_hsizes)
    elif integration == "euler":
        nssm_transition = CartPoleNSSMTransition(nx=4, nu=1, dt=Ts, hsizes=transition_hsizes)
    else:
        raise ValueError(f"unknown integration: {integration!r}")

    nssm_node = Node(
        nssm_transition,
        input_keys=["x0", "u_future"],
        output_keys=["x0"],  # recursive state key, same trick as train_nssm_sysid.py
        name="nssm_transition",
    )
    dynamics_system = System([nssm_node], nsteps=N_pred)

    def fy_fn(x0):
        return x0[:, :, [0, 2]]

    fy_node = Node(fy_fn, input_keys=["x0"], output_keys=["y_pred"], name="obs_extractor")

    y_pred = variable("y_pred")
    y_target = variable("y_target")
    x_pred = variable("x0")

    loss_pos = (y_pred[:, :, 0:1] == y_target[:, :, 0:1]) ^ 2
    loss_pos = loss_pos * 5.0
    loss_pos.update_name("loss_pos_mse")

    loss_theta = circular_theta_objective(
        y_pred[:, :, 1:2], y_target[:, :, 1:2], weight=1.0, name="loss_theta_circular"
    )

    x_curr = x_pred[:, 1:, :]
    x_prev = x_pred[:, :-1, :]
    loss_dx = (x_curr == x_prev) ^ 2
    loss_dx = loss_dx * Qdx
    loss_dx.update_name("loss_dx_smooth")

    pos_bound, theta_bound = y_bounds
    y_upper = torch.tensor([pos_bound, theta_bound])
    y_lower = -y_upper
    con_upper = (y_pred <= y_upper) ^ 2
    con_lower = (y_pred >= y_lower) ^ 2
    con_upper = con_upper * Qy
    con_lower = con_lower * Qy
    con_upper.update_name("loss_y_upper_bound")
    con_lower.update_name("loss_y_lower_bound")

    obj = PenaltyLoss([loss_pos, loss_theta, loss_dx], [con_upper, con_lower])
    problem = Problem(nodes=[dynamics_system, fy_node], loss=obj)
    return problem


def make_fullobs_dataset(num_samples, N_pred, Ts, F_max, batch_size, name, seed, signal_type="prbs"):
    """Reuses generate_cartpole_dataset (N_p doesn't matter here since no
    observer window is used -- set N_p=1, its y_past/u_past outputs are just
    unused/ignored downstream) but repackages x_target[:,0,:] as "x0" key
    (matching this script's node's expected input key) and drops the
    unused y_past/u_past/x_past keys from the dict.
    """
    dataset, loader = generate_cartpole_dataset(
        num_samples=num_samples, N_p=2, N_pred=N_pred, Ts=Ts, F_max=F_max,
        batch_size=batch_size, name=name, seed=seed, signal_type=signal_type,
    )
    dataset.datadict["x0"] = dataset.datadict["x_target"][:, 0:1, :].clone()
    for stale_key in ["y_past", "u_past", "x_past"]:
        if stale_key in dataset.datadict:
            del dataset.datadict[stale_key]
    from neuromancer.dataset import DictDataset
    from torch.utils.data import DataLoader
    new_dataset = DictDataset(dataset.datadict, name=name)
    shuffle = (name == "train")
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=new_dataset.collate_fn)
    return new_dataset, new_loader


if __name__ == "__main__":
    N_pred = 100
    Ts = 0.02
    batch_size = 32
    epochs = 400
    patience = 40
    Qdx = 0.01
    hsizes = [32, 32]
    integration = "euler"  # RK4도 시도해봤음 (theta RMSE 12.37->11.31deg, 소폭 개선) --
    # NODE(Neural ODE) 방향은 시간 관계상 이후 과제로 보류, 기본값은 euler로 유지
    signal_type = "prbs"

    print("1. 데이터셋 생성 중 (fully observable -- 진짜 x0 그대로 사용)...")
    train_dataset, train_loader = make_fullobs_dataset(
        4000, N_pred, Ts, F_max=5.0, batch_size=batch_size, name="train", seed=0, signal_type=signal_type,
    )
    val_dataset, val_loader = make_fullobs_dataset(
        800, N_pred, Ts, F_max=5.0, batch_size=800, name="dev", seed=1, signal_type=signal_type,
    )

    print("2. Fully-observable NSSM SysID Problem 구축 중...")
    problem = build_fullobs_sysid_problem(
        N_pred=N_pred, Ts=Ts, Qdx=Qdx, transition_hsizes=hsizes, integration=integration,
    )

    print("3. Optimizer 및 Trainer 설정 중...")
    optimizer = torch.optim.AdamW(problem.parameters(), lr=2e-3)
    trainer = Trainer(
        problem, train_loader, val_loader, optimizer=optimizer,
        epochs=epochs, patience=patience, warmup=5,
        train_metric="train_loss", dev_metric="dev_loss", eval_metric="dev_loss",
        callback=PrintDevLossCallback(),
    )

    print("학습 시작!")
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    ckpt_name = "sysid.pth"
    torch.save(problem.state_dict(), ckpt_name)
    print(f"\n학습 완료! '{ckpt_name}'로 최적 모델이 저장되었습니다.")
