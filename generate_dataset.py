import torch
import numpy as np
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset

from cartpole_ode import make_ground_truth_integrator, rollout


def generate_prbs(
    num_samples: int,
    total_steps: int,
    Ts: float,
    F_max: float = 5.0,
    min_hold_time: float = 0.1,
    max_hold_time: float = 0.5,
) -> torch.Tensor:
    """Amplitude-Modulated PRBS (APRS) control input sequence.

    Instead of just 3 discrete levels (-F_max, 0, +F_max), force values are
    sampled uniformly from [-F_max, +F_max] for each hold period.
    This provides continuous force excitation for better nonlinear identification.

    num_samples: number of trajectories to generate
    total_steps: total number of timesteps
    Ts: sampling time [s], used to convert hold time (s) -> hold length (steps)
    F_max: max control force [N]
    min_hold_time, max_hold_time: min/max duration [s] a level is held
    returns: shape (num_samples, total_steps, 1)
    """
    min_hold = max(1, round(min_hold_time / Ts))
    max_hold = max(min_hold, round(max_hold_time / Ts))

    u_seq = torch.zeros(num_samples, total_steps, 1)

    for i in range(num_samples):
        step = 0
        while step < total_steps:
            hold_len = np.random.randint(min_hold, max_hold + 1)
            val = np.random.uniform(-F_max, F_max)
            end_step = min(step + hold_len, total_steps)
            u_seq[i, step:end_step, 0] = float(val)
            step = end_step

    return u_seq


def generate_chirp(
    num_samples: int,
    total_steps: int,
    Ts: float,
    F_max: float = 5.0,
    f_low: float = 0.1,
    f_high: float = 3.0,
) -> torch.Tensor:
    """Linear frequency sweep (chirp) control input:
        u(t) = F_max * sin(2*pi*f(t)*t),  f(t) = f_low + (f_high-f_low)*t/T
    Sweeps from f_low to f_high [Hz] over the trajectory duration, covering
    the system's frequency range continuously (unlike PRBS's step changes) --
    classic system-ID excitation signal, good for probing frequency response
    across a band in one shot. f_high=3Hz stays well under the Ts=0.02s
    (50Hz sampling, Nyquist=25Hz) limit.

    Each trajectory gets a random phase offset and sweep DIRECTION (up or
    down) so num_samples trajectories aren't all identical.
    returns: shape (num_samples, total_steps, 1)
    """
    T_total = total_steps * Ts
    t = torch.arange(total_steps, dtype=torch.float32) * Ts  # (total_steps,)

    u_seq = torch.zeros(num_samples, total_steps, 1)
    for i in range(num_samples):
        phase0 = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.5:
            f_t = f_low + (f_high - f_low) * (t / T_total)
        else:
            f_t = f_high - (f_high - f_low) * (t / T_total)  # sweep down instead
        # instantaneous phase = integral of 2*pi*f(t) dt (trapezoidal, since f_t is linear in t this is exact)
        phase = phase0 + 2 * np.pi * torch.cumsum(f_t, dim=0) * Ts
        u_seq[i, :, 0] = F_max * torch.sin(phase)
    return u_seq


def generate_multisine(
    num_samples: int,
    total_steps: int,
    Ts: float,
    F_max: float = 5.0,
    n_tones: int = 6,
    f_low: float = 0.1,
    f_high: float = 3.0,
) -> torch.Tensor:
    """Sum-of-sines (multisine) control input:
        u(t) = A * sum_k sin(2*pi*f_k*t + phi_k),  f_k ~ U(f_low, f_high)
    A classic broadband system-ID excitation -- unlike a chirp (one frequency
    at a time), a multisine excites several frequencies SIMULTANEOUSLY at
    every instant, which can reveal nonlinear cross-frequency coupling a
    chirp might miss. Amplitude normalized (avg over tones) to stay near
    +-F_max like the other signal types, then clipped as a safety margin.

    Each trajectory gets its own random set of n_tones frequencies/phases.
    returns: shape (num_samples, total_steps, 1)
    """
    t = torch.arange(total_steps, dtype=torch.float32) * Ts  # (total_steps,)
    u_seq = torch.zeros(num_samples, total_steps, 1)
    for i in range(num_samples):
        freqs = np.random.uniform(f_low, f_high, size=n_tones)
        phases = np.random.uniform(0, 2 * np.pi, size=n_tones)
        signal = torch.zeros(total_steps)
        for f_k, phi_k in zip(freqs, phases):
            signal += torch.sin(2 * np.pi * f_k * t + phi_k)
        signal = signal / n_tones  # normalize so tones' sum stays near unit amplitude
        u_seq[i, :, 0] = torch.clamp(F_max * signal, -F_max, F_max)
    return u_seq


def _generate_raw(
    num_samples: int,
    N_p: int,
    N_pred: int,
    Ts: float,
    F_max: float,
    min_hold_time: float,
    max_hold_time: float,
    partial_obs: bool,
    pos_range: float = 1.0,
    vel_range: float = 1.0,
    theta_range: float = np.pi,
    theta_dot_range: float = 6.0,
    theta_center: float = 0.0,
    signal_type: str = "prbs",
) -> dict:
    """Generate raw (unsplit) tensors for num_samples trajectories.
    Same logic as before, factored out so train/dev/test splits can reuse it.

    x0 range parameters default to the original wide/swing-up-covering values
    (unchanged behavior for existing callers). Override them (e.g. narrow
    ranges + a gentler F_max) to generate a batch concentrated near a
    particular region, such as the origin -- see _generate_raw_mixed below,
    which was added because check_origin_data_density.py found essentially
    ZERO training samples with ||x||<0.3 in the default wide distribution.

    signal_type: "prbs" (default, unchanged), "chirp", "multisine", or
    "mixed" (splits num_samples roughly equally across all three -- see
    _generate_raw_mixed_signals below). min_hold_time/max_hold_time are
    ignored for chirp/multisine/mixed (PRBS-specific).
    """
    integrator = make_ground_truth_integrator(Ts=Ts)

    # Initial state x0 = [p, p_dot, theta, theta_dot], covering both balancing
    # and swing-up regions (theta sampled over full [-pi, pi]) by default.
    p0 = torch.FloatTensor(num_samples, 1).uniform_(-pos_range, pos_range)
    p_dot0 = torch.FloatTensor(num_samples, 1).uniform_(-vel_range, vel_range)
    theta0 = torch.FloatTensor(num_samples, 1).uniform_(-theta_range, theta_range) + theta_center
    theta_dot0 = torch.FloatTensor(num_samples, 1).uniform_(-theta_dot_range, theta_dot_range)
    x0 = torch.cat([p0, p_dot0, theta0, theta_dot0], dim=-1)

    total_steps = N_p + N_pred
    if signal_type == "prbs":
        u_seq = generate_prbs(
            num_samples, total_steps, Ts, F_max=F_max,
            min_hold_time=min_hold_time, max_hold_time=max_hold_time,
        )
    elif signal_type == "chirp":
        u_seq = generate_chirp(num_samples, total_steps, Ts, F_max=F_max)
    elif signal_type == "multisine":
        u_seq = generate_multisine(num_samples, total_steps, Ts, F_max=F_max)
    elif signal_type == "mixed":
        # Split num_samples roughly evenly across the three signal types and
        # concatenate their control sequences into one batch. x0 was already
        # sampled once above (shared across the whole batch, same as any
        # other signal_type) -- only the EXCITATION signal differs by chunk.
        n_prbs = num_samples // 3
        n_chirp = num_samples // 3
        n_multisine = num_samples - n_prbs - n_chirp  # remainder to multisine
        u_prbs = generate_prbs(n_prbs, total_steps, Ts, F_max=F_max,
                                min_hold_time=min_hold_time, max_hold_time=max_hold_time)
        u_chirp = generate_chirp(n_chirp, total_steps, Ts, F_max=F_max)
        u_multisine = generate_multisine(n_multisine, total_steps, Ts, F_max=F_max)
        u_seq = torch.cat([u_prbs, u_chirp, u_multisine], dim=0)
    else:
        raise ValueError(f"unknown signal_type: {signal_type!r} "
                          f"(expected 'prbs', 'chirp', 'multisine', or 'mixed')")

    x_full = rollout(integrator, x0, u_seq)  # (num_samples, total_steps+1, 4)

    if partial_obs:
        y_full = x_full[:, :, [0, 2]]  # [cart_pos, theta]
    else:
        y_full = x_full

    idx_curr = N_p - 1  # current time t_k (0-indexed)

    y_past = y_full[:, 0:N_p, :]
    x_past = x_full[:, 0:N_p, :]
    # u_past: the controls that produced the y_past window's state transitions.
    # y_past covers state indices 0..N_p-1 (N_p states); the control taking
    # state k -> k+1 is u_seq[k], so N_p states need N_p-1 controls
    # (u_seq[0]..u_seq[N_p-2]) to connect them. NOT the same as u_future[0]
    # (=u_seq[idx_curr]=u_seq[N_p-1]), which is the control about to be
    # applied FROM the current state onward.
    u_past = u_seq[:, 0:N_p - 1, :]
    u_future = u_seq[:, idx_curr: idx_curr + N_pred, :]
    y_target = y_full[:, idx_curr: idx_curr + N_pred + 1, :]
    x_target = x_full[:, idx_curr: idx_curr + N_pred + 1, :]

    return {
        "y_past": y_past,
        "u_past": u_past,
        "u_future": u_future,
        "y_target": y_target,
        "x_past": x_past,
        "x_target": x_target,
    }


def _generate_raw_mixed(
    num_samples: int,
    N_p: int,
    N_pred: int,
    Ts: float,
    F_max: float,
    min_hold_time: float,
    max_hold_time: float,
    partial_obs: bool,
    origin_frac: float = 0.3,
    origin_F_max: float = 1.0,
    origin_range: float = 0.3,
    signal_type: str = "prbs",
) -> dict:
    """Mix the default wide/swing-up-covering batch with an ORIGIN-FOCUSED
    batch: narrow x0 (+-origin_range on every state dim) AND a much gentler
    F_max (origin_F_max, default 1.0N vs the usual 5.0N) so those trajectories
    actually LINGER near the origin for many steps instead of being yanked
    away immediately by strong PRBS forcing.

    Motivated by check_origin_data_density.py finding essentially ZERO
    ||x||<0.3 samples in the default wide distribution -- the model was never
    trained on data near its own claimed equilibrium.

    origin_frac: fraction of num_samples drawn from the origin-focused batch
        (default 0.3 = 30%). The rest use the original wide distribution.
    """
    n_origin = int(round(num_samples * origin_frac))
    n_wide = num_samples - n_origin

    data_wide = _generate_raw(
        n_wide, N_p, N_pred, Ts, F_max, min_hold_time, max_hold_time, partial_obs,
        signal_type=signal_type,
    )  # default (wide) x0 ranges

    data_origin = _generate_raw(
        n_origin, N_p, N_pred, Ts, origin_F_max, min_hold_time, max_hold_time, partial_obs,
        pos_range=origin_range, vel_range=origin_range,
        theta_range=origin_range, theta_dot_range=origin_range,
        signal_type=signal_type,
    )

    return {k: torch.cat([data_wide[k], data_origin[k]], dim=0) for k in data_wide}


def generate_cartpole_dataset(
    num_samples: int = 1000,
    N_p: int = 5,
    N_pred: int = 50,
    Ts: float = 0.02,
    F_max: float = 5.0,
    min_hold_time: float = 0.1,
    max_hold_time: float = 0.5,
    partial_obs: bool = True,
    batch_size: int = 64,
    seed: int = None,
    name: str = "train",
    origin_frac: float = 0.0,
    origin_F_max: float = 1.0,
    origin_range: float = 0.3,
    signal_type: str = "prbs",
):
    """Generate a single Neuromancer DictDataset + DataLoader (no train/val/test split).

    origin_frac: if > 0, mixes in an origin-focused batch (see
        _generate_raw_mixed) covering that fraction of num_samples. Default 0
        = unchanged original behavior (100% wide distribution).
    signal_type: "prbs" (default), "chirp", or "multisine" -- see _generate_raw.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if origin_frac > 0:
        data_dict = _generate_raw_mixed(
            num_samples, N_p, N_pred, Ts, F_max, min_hold_time, max_hold_time, partial_obs,
            origin_frac=origin_frac, origin_F_max=origin_F_max, origin_range=origin_range,
            signal_type=signal_type,
        )
    else:
        data_dict = _generate_raw(
            num_samples, N_p, N_pred, Ts, F_max, min_hold_time, max_hold_time, partial_obs,
            signal_type=signal_type,
        )
    dataset = DictDataset(data_dict, name=name)
    shuffle = (name == "train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
    return dataset, dataloader


def generate_cartpole_datasets(
    num_samples: int = 1000,
    splits: tuple = (0.7, 0.15, 0.15),
    N_p: int = 5,
    N_pred: int = 50,
    Ts: float = 0.02,
    F_max: float = 5.0,
    min_hold_time: float = 0.1,
    max_hold_time: float = 0.5,
    partial_obs: bool = True,
    batch_size: int = 64,
    seed: int = 0,
):
    """Generate train/dev/test DictDatasets + DataLoaders from the SAME underlying distribution."""
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dict = _generate_raw(
        num_samples, N_p, N_pred, Ts, F_max, min_hold_time, max_hold_time, partial_obs,
    )

    n_train = int(round(splits[0] * num_samples))
    n_dev = int(round(splits[1] * num_samples))
    n_test = num_samples - n_train - n_dev

    perm = torch.randperm(num_samples)
    idx_train = perm[:n_train]
    idx_dev = perm[n_train:n_train + n_dev]
    idx_test = perm[n_train + n_dev:]

    out = {}
    for name, idx in [("train", idx_train), ("dev", idx_dev), ("test", idx_test)]:
        split_dict = {k: v[idx] for k, v in data_dict.items()}
        dataset = DictDataset(split_dict, name=name)
        shuffle = (name == "train")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
        out[name] = (dataset, dataloader)

    print(f"Split sizes -> train: {n_train}, dev: {n_dev}, test: {n_test}")
    return out


if __name__ == "__main__":
    splits = generate_cartpole_datasets(
        num_samples=1000, splits=(0.7, 0.15, 0.15),
        N_p=5, N_pred=50, Ts=0.02, F_max=5.0,
        min_hold_time=0.1, max_hold_time=0.5, seed=0,
    )

    for name, (dataset, dataloader) in splits.items():
        sample = next(iter(dataloader))
        print(f"--- {name} batch ---")
        print("y_past   :", sample["y_past"].shape)
        print("u_future :", sample["u_future"].shape)
        print("y_target :", sample["y_target"].shape)