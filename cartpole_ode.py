"""
Cart-pole ground-truth dynamics as a Neuromancer ODESystem.

This is the "real system" used to generate data for system identification.
Its physical parameters are fixed (registered as buffers, NOT nn.Parameter),
so it is never trained -- it only ever produces data. The identified model
(a separate NSSM/BN-SSM, built elsewhere) is what gets fit to that data.

State convention (must match dynamics.py, the numpy side-validation model):
    x = [cart_pos, cart_vel, theta, theta_dot]
    theta = 0   -> pole pointing UP    (unstable equilibrium)
    theta = pi  -> pole pointing DOWN  (stable equilibrium)
    theta measured COUNTER-CLOCKWISE from upward vertical.

Input:
    u = force on the cart [N], positive = +x (right).

Equations (CCW-positive theta, viscous cart-ground friction b):
    temp       = (u - b*x_dot - m_p*l*theta_dot^2*sin(theta)) / (m_c+m_p)
    theta_ddot = (g*sin(theta) + cos(theta)*temp) /
                 (l * (4/3 - m_p*cos(theta)^2 / (m_c+m_p)))
    x_ddot     = temp + m_p*l*theta_ddot*cos(theta) / (m_c+m_p)
"""

import torch
import torch.nn as nn
from neuromancer.dynamics.ode import ODESystem
from neuromancer.dynamics import integrators


class CartPoleGroundTruth(ODESystem):
    """RHS of the cart-pole ODE: dx/dt = f(x, u).

    insize = nx + nu = 4 + 1 = 5
    outsize = nx = 4
    """

    def __init__(self, insize=5, outsize=4,
                 m_c=1.0, m_p=0.1, l=0.5, g=9.8, b=0.1):
        super().__init__(insize=insize, outsize=outsize)
        # Fixed physical constants -- buffers, not nn.Parameter, so autograd
        # never sees them as trainable and .parameters() is empty.
        self.register_buffer("m_c", torch.tensor(float(m_c)))
        self.register_buffer("m_p", torch.tensor(float(m_p)))
        self.register_buffer("l", torch.tensor(float(l)))
        self.register_buffer("g", torch.tensor(float(g)))
        self.register_buffer("b", torch.tensor(float(b)))

    def ode_equations(self, x, u):
        """
        x: (batch, 4) = [cart_pos, cart_vel, theta, theta_dot]
        u: (batch, 1) = force
        returns dx/dt: (batch, 4)
        """
        pos = x[:, 0:1]
        vel = x[:, 1:2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:4]

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        total_mass = self.m_c + self.m_p

        temp = (u - self.b * vel - self.m_p * self.l * theta_dot ** 2 * sin_t) / total_mass
        theta_ddot = (self.g * sin_t + cos_t * temp) / (
            self.l * (4.0 / 3.0 - self.m_p * cos_t ** 2 / total_mass)
        )
        x_ddot = temp + self.m_p * self.l * theta_ddot * cos_t / total_mass

        return torch.cat([vel, x_ddot, theta_dot, theta_ddot], dim=-1)


class CartPoleGroundTruthPinned(ODESystem):
    """RHS when the cart is rigidly pinned against the track wall (x, x_dot
    held fixed by the wall's reaction force). Derived by imposing x_ddot=0 in
    the coupled Lagrangian equations of motion -- NOT by naively zeroing u in
    the free-space formula (verified numerically these differ; see
    dynamics.py's dynamics_pinned for the derivation/cross-check):

        theta_ddot = (3*g)/(4*l) * sin(theta)   -- uniform-rod pendulum,
        matching the 4/3 rotational inertia factor used in CartPoleGroundTruth
        (NOT the point-mass formula g/l).

    u has NO effect on theta_ddot while pinned -- entirely absorbed by the
    wall. Still takes u as an input (ignored) so it has the same (x,u)->dx/dt
    interface as CartPoleGroundTruth and can be dropped into integrators.RK4
    identically.
    """

    def __init__(self, insize=5, outsize=4, m_c=1.0, m_p=0.1, l=0.5, g=9.8, b=0.1):
        super().__init__(insize=insize, outsize=outsize)
        self.register_buffer("l", torch.tensor(float(l)))
        self.register_buffer("g", torch.tensor(float(g)))

    def ode_equations(self, x, u):
        theta = x[:, 2:3]
        theta_dot = x[:, 3:4]
        theta_ddot = (3.0 * self.g) / (4.0 * self.l) * torch.sin(theta)
        zeros = torch.zeros_like(theta)
        return torch.cat([zeros, zeros, theta_dot, theta_ddot], dim=-1)


def make_ground_truth_integrator(Ts=0.02, **physics_kwargs):
    """Convenience factory: returns an RK4 integrator wrapping the ground-truth ODE.

    Usage:
        integrator = make_ground_truth_integrator(Ts=0.02)
        x_next = integrator(x, u)   # x: (batch,4), u: (batch,1) -> (batch,4)
    """
    ode = CartPoleGroundTruth(**physics_kwargs)
    return integrators.RK4(ode, h=Ts)


def make_pinned_integrator(Ts=0.02, **physics_kwargs):
    """Same as make_ground_truth_integrator but for the wall-pinned dynamics
    (see CartPoleGroundTruthPinned). u is still accepted for interface
    consistency but has no effect on the output."""
    ode = CartPoleGroundTruthPinned(**physics_kwargs)
    return integrators.RK4(ode, h=Ts)


def apply_track_limits(x: torch.Tensor, track_limit: float) -> torch.Tensor:
    """Inelastic hard stop at the rail ends, batched: clip cart_pos, kill ONLY
    the outward velocity component. Mirrors dynamics.py's numpy version
    exactly so both simulators represent the same physical rig.

    x: (batch, 4) = [cart_pos, cart_vel, theta, theta_dot]
    """
    pos = x[:, 0:1]
    vel = x[:, 1:2]

    over_pos = pos > track_limit
    under_neg = pos < -track_limit

    pos = torch.where(over_pos, torch.full_like(pos, track_limit), pos)
    pos = torch.where(under_neg, torch.full_like(pos, -track_limit), pos)

    kill_outward_pos = over_pos & (vel > 0)
    kill_outward_neg = under_neg & (vel < 0)
    vel = torch.where(kill_outward_pos | kill_outward_neg, torch.zeros_like(vel), vel)

    return torch.cat([pos, vel, x[:, 2:3], x[:, 3:4]], dim=-1)


@torch.no_grad()
def rollout(integrator, x0: torch.Tensor, u_seq: torch.Tensor, track_limit: float = None,
            use_wall_physics: bool = True) -> torch.Tensor:
    """Roll out a batch of trajectories under the ground-truth dynamics.

    x0:    (batch, 4)      initial states
    u_seq: (batch, T, 1)   force sequence (zero-order hold per step)
    returns states: (batch, T+1, 4), states[:,0] == x0
    Track limits (hard stop at +-track_limit) applied after every step, matching
    dynamics.py's simulate(). Set track_limit=None to disable (e.g. for
    cross-checking against pre-track-limit numpy runs).

    use_wall_physics (default True): when a sample is AT the wall and being
    pushed further into it, use the physically-correct wall-contact dynamics
    for that step -- u has no effect on theta while pinned, the wall reaction
    absorbs it (see dynamics.py's dynamics_pinned derivation). Per-sample
    masked via torch.where since different batch elements may be pinned or
    free at the same step. Set False only for backward-compat / comparing
    against the old (unphysical) behavior -- not recommended otherwise.
    """
    batch, T, _ = u_seq.shape
    states = torch.zeros(batch, T + 1, x0.shape[-1], dtype=x0.dtype, device=x0.device)
    x = apply_track_limits(x0, track_limit) if track_limit is not None else x0
    states[:, 0] = x

    pinned_integrator = None
    if use_wall_physics and track_limit is not None:
        pinned_integrator = make_pinned_integrator(Ts=float(integrator.h))

    for k in range(T):
        u = u_seq[:, k]
        if pinned_integrator is not None:
            pos = x[:, 0:1]
            at_right = pos >= (track_limit - 1e-6)
            at_left = pos <= (-track_limit + 1e-6)
            pinned_mask = (at_right & (u > 0)) | (at_left & (u < 0))  # (batch,1)

            x_free = integrator(x, u)
            x_pinned = pinned_integrator(x, u)
            x = torch.where(pinned_mask, x_pinned, x_free)
        else:
            x = integrator(x, u)
        if track_limit is not None:
            x = apply_track_limits(x, track_limit)
        states[:, k + 1] = x
    return states
