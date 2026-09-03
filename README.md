# Cart-Pole Differentiable Predictive Control (DPC)

Cart-pole balancing and swing-up control using Differentiable Predictive
Control (DPC), following the [neuromancer](https://github.com/pnnl/neuromancer)
framework: a neural network policy is trained offline through a learned
(system-identified) dynamics model, then deployed with zero online
optimization (a single forward pass per control step) -- in contrast to
online MPC, which re-solves an optimization problem every step.

## Pipeline

```
1. Ground truth simulator      cartpole_ode.py
2. Data generation             generate_dataset.py   (PRBS / chirp / multisine)
3. System identification       train_nssm_sysid_fullobs.py  ->  sysid.pth
4. Balancing policy             train_dpc_balance_curriculum.py  ->  policy_balance.pth
5. Swing-up policy               train_dpc_swingup.py  ->  policy_swingup.pth
```

**Design note (fully observable):** an earlier version of this project used
partial observation (a neural observer estimating full state from a window
of past position/angle measurements). This was found to work poorly and was
removed -- the current pipeline assumes the full state
`[cart_pos, cart_vel, theta, theta_dot]` is directly measured. See
`train_nssm_sysid_fullobs.py`'s docstring for the reasoning. A physics-based
observer (e.g. an Extended Kalman Filter) is a natural next step, left as
future work.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the dynamics model (system identification)
python train_nssm_sysid_fullobs.py
python eval_sysid_fullobs.py --ckpt sysid.pth

# 2. Train the balancing policy (curriculum: theta0 +-0.05 -> +-0.3 rad)
python train_dpc_balance_curriculum.py
python eval_balance_policy.py --ckpt policy_balance.pth

# 3. Train the swing-up policy (warm-started from the balancing policy)
python train_dpc_swingup.py
python eval_balance_policy.py --ckpt policy_swingup.pth \
    --theta_center 0 --theta_range 3.14159 --theta_dot_range 1.0
```

## Key results (ground truth, 200 initial conditions)

| | Balancing | Swing-up |
|---|---|---|
| Success rate | 98.5% | 99.5% (wrapped) |
| Final \|theta\| mean | 1.62 deg | 0.67 deg |
| Initial condition range | theta0 +- 0.3 rad | theta0 +- pi (full range) |

DPC vs online QP-MPC (same test distribution): DPC matches or exceeds MPC's
success rate while being ~20-500x faster per control step (a single MLP
forward pass, ~0.1ms, vs re-solving a QP online). See `mpc_baselines.py`.

## File overview

**Core pipeline**
- `cartpole_ode.py` -- ground-truth nonlinear ODE (RK4), used both to
  generate training data and to evaluate policies in closed loop.
- `generate_dataset.py` -- excitation signal generation (PRBS / chirp /
  multisine) and rollout-based dataset construction.
- `nssm_model.py` -- shared NSSM (Neural State Space Model) transition
  network classes, used by both the sysID training script and the frozen
  dynamics model loaded during policy training.
- `train_nssm_sysid_fullobs.py` / `eval_sysid_fullobs.py` -- system
  identification (fully observable) and its evaluation.
- `dpc_policy_balance.py` -- policy network definition and the closed-loop
  DPC training graph (shared by balancing and swing-up).
- `train_dpc_balance_curriculum.py` -- balancing policy training (curriculum
  over initial condition range).
- `train_dpc_swingup.py` -- swing-up policy training (warm-started from the
  balancing policy; uses an energy-matching loss term in addition to
  state-tracking, needed to escape the pole's downward stable equilibrium).
- `eval_balance_policy.py` -- shared evaluation script for both balancing
  and swing-up policies (ground truth vs identified-model rollouts).

**MPC baselines**
- `mpc_baselines.py` -- linear MPC (QP, solved via OSQP) baselines, both
  fixed-linearization (linearize once at the origin) and re-linearized
  (successive linearization at every step), compared against DPC.

**Wall-contact case study**
A diagnostic deep-dive into the ~6.5% of test cases where the balancing
policy's cart approaches the track limit: is this a genuine physical
limitation (given the actuator force limit) or a controller/method
limitation? Conclusion: mostly a method limitation -- DPC solves 10/13 of
these hard cases; every linear-model-based baseline tried (fixed MPC,
re-linearized MPC, LQR) solves 0/13.
- `extract_wall_contact_cases.py` -- identifies and saves the hard cases.
- `verify_dpc_on_wall_cases.py`, `test_wall_cases_long_horizon_mpc.py`,
  `test_wall_cases_relin_mpc.py`, `test_wall_cases_lqr.py` -- re-test those
  same cases with DPC / MPC (various horizons) / re-linearized MPC / LQR.

## Known limitations

- Fully observable assumption (see Design note above) -- not validated for
  partial observation / sensor noise.
- Swing-up's "success" metric uses angle-wrapped theta -- a small fraction
  of trajectories reach upright after a full extra rotation, which is
  physically equivalent but may not be desirable in a real deployment
  (unbounded cable wind-up, etc.).
- Some initial-condition chattering remains in the swing-up policy's control
  input during the first ~2s (energy-pumping phase).
- Hyperparameters (loss weights, energy term weight) were tuned empirically,
  not through a systematic search.
