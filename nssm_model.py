"""Shared NSSM transition model components -- used by BOTH the sysID training
script (train_nssm_sysid_fullobs.py) and the policy pipeline
(dpc_policy_balance.py, which loads a frozen trained transition model).

Split out into its own module because these classes must NOT live inside a
throwaway training-script entry point -- both consumers need to import them
independently, and neither should depend on the other's __main__ block.

Fully-observable design: no observer here (see project decision to drop the
partial-observability / neural observer approach entirely -- see README).
"""
import torch
import torch.nn as nn

from neuromancer.modules import blocks
from neuromancer.constraint import Variable
from neuromancer.dynamics import integrators
from neuromancer.callbacks import Callback


class PrintDevLossCallback(Callback):
    """Trainer's default print only shows train_loss (when logger=None).
    Print dev_loss too at end of each eval so plateau/best-so-far is visible.
    """
    def end_eval(self, trainer, output):
        key = f"mean_{trainer.dev_metric}"
        if key in output:
            print(f"           dev_loss: {output[key].item():.6f}  "
                  f"(best so far: {trainer.best_devloss.item():.6f}, badcount: {trainer.badcount})")


def sincos_embed_state(x):
    """[pos, vel, theta, theta_dot] (batch,4) -> [pos, vel, sin(theta), cos(theta), theta_dot] (batch,5)
    Network-input-only transform: keeps theta periodic-aware and bounded to
    [-1,1] regardless of how many full rotations it has accumulated, without
    changing the underlying recursive state representation (which stays raw
    theta, so swing-counting / long trajectories still make physical sense).
    """
    pos = x[:, 0:1]
    vel = x[:, 1:2]
    theta = x[:, 2:3]
    theta_dot = x[:, 3:4]
    return torch.cat([pos, vel, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)


def circular_theta_objective(theta_pred_var, theta_target_var, weight=1.0, name=None):
    """Circular (wrap-aware) squared-error loss for an angle variable, built as
    2*(1 - cos(pred - target)). Equals the true squared angular distance for
    small errors (1-cos(d) ~ d^2/2) but stays smooth and periodic for large
    errors, unlike raw (pred-target)^2 which spuriously penalizes physically-
    near angles that differ by ~2*pi (e.g. +3.14 vs -3.14) as if ~2*pi apart.
    """
    diff = theta_pred_var - theta_target_var  # Variable
    circ = Variable(input_variables=[diff], func=lambda d: 2.0 * (1.0 - torch.cos(d)),
                     display_name="circular_dist")
    obj = circ.minimize(metric=torch.mean, weight=weight, name=name)
    if name is not None:
        obj.output_keys = [name]
    return obj


class CartPoleNSSMTransition(nn.Module):
    """NSSM discrete-time state transition (Euler): x_{k+1} = x_k + dt * f_theta(embed(x_k), u_k)
    Net input is sincos_embed_state(x) (theta -> [sin,cos]) concatenated with u (6-dim total).
    Output (dx/dt) stays raw 4-dim -- only the network's INPUT representation changes.
    """

    def __init__(self, nx: int = 4, nu: int = 1, dt: float = 0.02, hsizes: list = [60, 60]):
        super().__init__()
        self.dt = dt
        self.net = blocks.MLP(
            insize=nx + 1 + nu,  # sincos embed adds 1 dim: theta -> sin,cos
            outsize=nx,
            hsizes=hsizes,
            nonlin=nn.GELU,
            linear_map=nn.Linear,
        )

    def forward(self, x, u):
        x_embed = sincos_embed_state(x)
        xu = torch.cat([x_embed, u], dim=-1)
        dx = self.net(xu)
        return x + self.dt * dx


class CartPoleFRHS(nn.Module):
    """f_theta(x,u) ~= dx/dt, sin/cos-embedded input, for RK4 integration
    (raw dx/dt output, not a discrete state update)."""

    def __init__(self, nx: int = 4, nu: int = 1, hsizes: list = [60, 60]):
        super().__init__()
        self.net = blocks.MLP(
            insize=nx + 1 + nu,
            outsize=nx,
            hsizes=hsizes,
            nonlin=nn.GELU,
            linear_map=nn.Linear,
        )
        # neuromancer's Integrator.__init__ reads block.in_features/out_features
        # as metadata (not used inside RK4.integrate() itself) -- must exist.
        self.in_features = nx
        self.out_features = nx

    def forward(self, x, u):
        x_embed = sincos_embed_state(x)
        xu = torch.cat([x_embed, u], dim=-1)
        return self.net(xu)


class CartPoleNSSMTransitionRK4(nn.Module):
    """NSSM discrete-time transition via RK4 integration of a learned
    continuous-time f_theta(x,u) ~= dx/dt. Matches ground truth's own
    integration scheme (RK4), unlike the Euler version above.
    """

    def __init__(self, nx: int = 4, nu: int = 1, dt: float = 0.02, hsizes: list = [60, 60]):
        super().__init__()
        f_theta = CartPoleFRHS(nx=nx, nu=nu, hsizes=hsizes)
        self.integrator = integrators.RK4(f_theta, h=dt)

    def forward(self, x, u):
        return self.integrator(x, u)
