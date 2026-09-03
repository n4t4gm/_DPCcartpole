"""Differentiable Predictive Control: balancing policy for cart-pole.

Trains a neural feedback policy pi_theta(x) -> u on top of the FROZEN,
already-trained NSSM dynamics model (sysid.pth).
Only the policy is trained; the dynamics model's parameters are frozen.

Simplification (first pass): the policy sees the full state x = [pos, vel,
theta, theta_dot] directly (assumes a known/given initial state), rather than
going through the partial-observation + observer chain used for sysID. This
keeps the closed-loop training graph simple; the observer can be added back
in later to make the whole pipeline end-to-end from raw sensor readings.

Closed-loop rollout pattern (matches JPC2022 Algorithm 1):
    x_k --policy--> u_k --dynamics--> x_{k+1} --policy--> u_{k+1} --> ...
Built with neuromancer System([policy_node, dynamics_node], nsteps=N):
System.forward slices data[k][:, i] per step for every node, and appends
each node's 2D output back as a new time-slice (see System.cat) -- so a
policy_node with input 'x' / output 'u', followed by a dynamics_node with
input 'x','u' / output 'x', naturally alternates policy/dynamics steps.
"""
import torch
import torch.nn as nn
import numpy as np

from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.constraint import variable, Variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.callbacks import Callback

from nssm_model import CartPoleNSSMTransition, sincos_embed_state, circular_theta_objective, PrintDevLossCallback
from cartpole_ode import make_ground_truth_integrator, rollout as gt_rollout

F_MAX = 5.0        # actuator limit [N], MUST match sysID data generation
TRACK_LIMIT = 1.5  # rail half-length [m], MUST match ground truth track limit
Ts = 0.02


class CartPolePolicy(nn.Module):
    """pi_theta(x) -> u, bounded to [-F_max, F_max] via tanh.
    Input is sincos-embedded (same reasoning as the dynamics model: theta can
    be many multiples of 2*pi away from the training distribution's "typical"
    range, and the policy needs a periodic-aware representation too)."""

    def __init__(self, nx: int = 4, F_max: float = F_MAX, hsizes: list = [60, 60]):
        super().__init__()
        self.F_max = F_max
        self.net = blocks.MLP(
            insize=nx + 1,  # sincos embed: theta -> sin,cos adds 1 dim
            outsize=1,
            hsizes=hsizes,
            nonlin=nn.GELU,
            linear_map=nn.Linear,
        )

    def forward(self, x):
        x_embed = sincos_embed_state(x)
        raw = self.net(x_embed)
        return self.F_max * torch.tanh(raw)


def load_frozen_dynamics(ckpt_path, Ts=Ts, hsizes=[32, 32]):
    """Load the trained NSSM transition module and freeze its parameters.
    Fully-observable sysID Problem's node list is [dynamics_system, fy_node]
    (no observer) -- transition params live under nodes.0.nodes.0.callable.
    """
    transition = CartPoleNSSMTransition(nx=4, nu=1, dt=Ts, hsizes=hsizes)
    full_state_dict = torch.load(ckpt_path, map_location="cpu")
    prefix = "nodes.0.nodes.0.callable."
    transition_state = {k[len(prefix):]: v for k, v in full_state_dict.items() if k.startswith(prefix)}
    missing, unexpected = transition.load_state_dict(transition_state, strict=True)
    for p in transition.parameters():
        p.requires_grad_(False)
    transition.eval()
    return transition


def build_balance_problem(transition, N: int = 80, Q=(8.0, 0.1, 10.0, 0.1), R=0.01, Qc=1.0,
                           policy=None, time_weight_power=None, energy_weight=None):
    """Closed-loop policy training graph.

    N: rollout horizon (steps). Kept within the dynamics model's trusted
       region (it was trained/evaluated well up to ~100 steps=2s).
    Q: per-state tracking weights [pos, vel, theta, theta_dot], reference=0.
    R: control effort weight.
    Qc: track-limit soft constraint weight.
    policy: reuse an existing CartPolePolicy instance (for curriculum
        warm-starting) instead of creating a fresh one.
    time_weight_power: if set (e.g. 4), the state-tracking loss is scaled by
        (t/N)**power at each timestep t -- near 0 early in the rollout, ramping
        to 1 by the end. This lets the policy swing widely (large transient
        deviation from upright/centered) during an early "energy pumping"
        phase without being penalized for it, while still being pushed hard to
        end up stabilized near x=0 by the final steps. None (default) keeps
        the original uniform-weight behavior used for balancing.
    energy_weight: if set, adds a mechanical-energy-matching loss
        Q_e * (E(theta,theta_dot) - E_upright)^2 for the pole, where
        E = 0.5*m_p*l^2*theta_dot^2 + m_p*g*l*cos(theta). theta=pi (hanging
        down) is a stable equilibrium of the FREE (u=0) pendulum -- purely
        distance-based tracking loss (loss_state above) gives the optimizer
        an easy, low-effort "settle near pi and stop" local minimum to fall
        into instead of the harder swing-up solution (empirically observed:
        swing-up policies converge to oscillating/resting near +-pi instead
        of reaching upright). The energy term is explicitly minimized ONLY
        at the upright energy level regardless of angle, so "resting at pi"
        (which has LOWER energy than upright, not equal) is no longer a
        free/attractive local minimum for this term -- it directly rewards
        "has enough energy to reach upright", independent of loss_state's
        distance-based signal. None (default) leaves it out entirely
        (unchanged behavior for balancing).
    """
    if policy is None:
        policy = CartPolePolicy(nx=4, F_max=F_MAX)
    policy_node = Node(policy, input_keys=["x"], output_keys=["u"], name="policy")

    # NOTE: previously wrapped in ClippedTransition (hard wall clip) -- removed
    # per the switch to a wall-less design (no physical track limit anywhere;
    # position is only softly discouraged from wandering via the Qc penalty
    # below, matching Amos et al./KCPO's approach of no explicit state
    # constraint, only cost-based tracking).
    dynamics_node = Node(transition, input_keys=["x", "u"], output_keys=["x"], name="dynamics")

    closed_loop = System([policy_node, dynamics_node], nsteps=N)

    x = variable("x")
    u = variable("u")

    Q_t = torch.tensor(Q).view(1, 1, 4)

    # theta(index 2)는 circular loss로 분리 -- raw quadratic (theta-0)^2은
    # theta=pi 근처에서 어느 쪽으로 살짝 기울었는지에 따라 gradient가
    # 불연속적으로 튀는 문제가 있음(mod-wrap과 동일한 종류의 문제, 수치로
    # 확인: d=pi-0.01일 때 gradient=+6.26, d=pi+0.01일 때 -6.26).
    # 에너지 항(cos 기반, 이미 매끄러움)과의 일관성을 위해서도 circular가 맞음.
    non_theta_idx = [0, 1, 3]  # pos, vel, theta_dot -- raw quadratic 그대로 유지
    Q_non_theta = Q_t[:, :, non_theta_idx]
    x_non_theta = x[:, :, non_theta_idx]
    x_weighted = x_non_theta * torch.sqrt(Q_non_theta)

    theta_var = x[:, :, 2:3]
    zero_target = theta_var * 0.0  # target=0 (upright), same shape/dtype as theta_var

    if time_weight_power is not None:
        t_idx = torch.arange(N + 1, dtype=torch.float32).view(1, N + 1, 1)
        time_w = (t_idx / N) ** time_weight_power  # (1, N+1, 1), 0 early -> 1 late
        x_weighted = x_weighted * torch.sqrt(time_w)  # sqrt so squaring gives exactly time_w scaling
        # NOTE: circular_theta_objective doesn't support a time-varying weight
        # multiplier directly; time_weight_power is unused with swing-up
        # currently (only tried once, reverted) so leaving this uncombined.
    # weighted quadratic-to-zero loss for non-theta states: sum_i Q_i * x_i^2
    loss_nontheta = (x_weighted == 0.0) ^ 2
    loss_nontheta.update_name("loss_state_tracking_nontheta")

    loss_theta_state = circular_theta_objective(theta_var, zero_target, weight=Q[2],
                                                 name="loss_theta_state_circular")

    loss_u = (u * np.sqrt(R) == 0.0) ^ 2
    loss_u.update_name("loss_control_effort")

    con_upper = (x[:, :, 0:1] <= TRACK_LIMIT) ^ 2
    con_lower = (x[:, :, 0:1] >= -TRACK_LIMIT) ^ 2
    con_upper = con_upper * Qc
    con_lower = con_lower * Qc
    con_upper.update_name("loss_track_upper")
    con_lower.update_name("loss_track_lower")

    objectives = [loss_nontheta, loss_theta_state, loss_u]
    if energy_weight is not None:
        theta_var = x[:, :, 2:3]
        theta_dot_var = x[:, :, 3:4]
        m_p, l_pole, g_grav = 0.1, 0.5, 9.8  # MUST match dynamics.py/cartpole_ode.py physics params
        E_upright = m_p * g_grav * l_pole

        def _energy_error(th, thd):
            E = 0.5 * m_p * (l_pole ** 2) * thd ** 2 + m_p * g_grav * l_pole * torch.cos(th)
            return E - E_upright

        energy_err_var = Variable(input_variables=[theta_var, theta_dot_var],
                                   func=_energy_error, display_name="energy_error")
        loss_energy = (energy_err_var == 0.0) ^ 2
        loss_energy = loss_energy * energy_weight
        loss_energy.update_name("loss_energy")
        objectives.append(loss_energy)

    obj = PenaltyLoss(objectives, [con_upper, con_lower])
    problem = Problem(nodes=[closed_loop], loss=obj)
    return problem, policy


def sample_balance_x0(num_samples, seed=None,
                       pos_range=0.5, vel_range=0.5, theta_range=0.3, theta_dot_range=0.5,
                       theta_center=0.0):
    if seed is not None:
        torch.manual_seed(seed)
    pos0 = torch.FloatTensor(num_samples, 1).uniform_(-pos_range, pos_range)
    vel0 = torch.FloatTensor(num_samples, 1).uniform_(-vel_range, vel_range)
    theta0 = torch.FloatTensor(num_samples, 1).uniform_(-theta_range, theta_range) + theta_center
    theta_dot0 = torch.FloatTensor(num_samples, 1).uniform_(-theta_dot_range, theta_dot_range)
    return torch.cat([pos0, vel0, theta0, theta_dot0], dim=-1)


def make_balance_dataset(num_samples, N, seed=None, name="train", **x0_kwargs):
    x0 = sample_balance_x0(num_samples, seed=seed, **x0_kwargs)
    from neuromancer.dataset import DictDataset
    from torch.utils.data import DataLoader
    data = {"x": x0.unsqueeze(1)}  # (batch, 1, 4), matches System's expected 3D init
    dataset = DictDataset(data, name=name)
    loader = DataLoader(dataset, batch_size=num_samples if name != "train" else 64,
                         shuffle=(name == "train"), collate_fn=dataset.collate_fn)
    return dataset, loader


if __name__ == "__main__":
    N = 80  # 1.6s rollout horizon
    epochs = 100

    print("1. Frozen dynamics 로드 중...")
    transition = load_frozen_dynamics("sysid.pth")

    print("2. Balancing dataset 생성 중...")
    train_dataset, train_loader = make_balance_dataset(2000, N, seed=0, name="train")
    dev_dataset, dev_loader = make_balance_dataset(400, N, seed=1, name="dev")

    print("3. Closed-loop policy problem 구축 중...")
    problem, policy = build_balance_problem(transition, N=N)

    print("4. Optimizer/Trainer 설정 중...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)  # ONLY policy params
    trainer = Trainer(
        problem, train_loader, dev_loader, optimizer=optimizer,
        epochs=epochs, patience=20, warmup=5,
        train_metric="train_loss", dev_metric="dev_loss", eval_metric="dev_loss",
        callback=PrintDevLossCallback(),
    )

    print("5. 학습 시작!")
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    torch.save(policy.state_dict(), "policy_balance.pth")
    print("\n학습 완료! 'policy_balance.pth' 저장됨.")
