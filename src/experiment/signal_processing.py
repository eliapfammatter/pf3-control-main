"""
Signal Processing Utilities for PF3 Experiment Data.

Contains:
- RPM extraction from tachometer pulse signals
- Spectrogram computation with frequency normalization
- Signal alignment using RMS minimization
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann


def extract_rpm_from_tachometer(
    tach_signal: np.ndarray,
    sample_rate: float,
    resample_freq: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract RPM from tachometer pulse signal.

    Args:
        tach_signal: Raw tachometer signal (1 pulse per revolution)
        sample_rate: Sample rate of the tachometer signal [Hz]
        resample_freq: Output resample frequency [Hz] (default: 1 Hz)

    Returns:
        Tuple of (rpm_values, time_array, t_start):
        - rpm_values: Resampled RPM values
        - time_array: Time array relative to t_start=0
        - t_start: Start time offset [s]
    """
    n_samples = len(tach_signal)
    t_raw = np.arange(n_samples) / sample_rate

    # Detect rising edges
    threshold = (tach_signal.max() + tach_signal.min()) / 2
    above_thresh = tach_signal > threshold
    edges = np.diff(above_thresh.astype(int))
    pulse_indices = np.where(edges == 1)[0]
    pulse_times = t_raw[pulse_indices]

    # Compute instantaneous RPM from pulse intervals
    pulse_intervals = np.diff(pulse_times)
    instant_rpm = 60.0 / pulse_intervals
    t_instant = pulse_times[:-1]

    # Resample to uniform grid
    dt = 1.0 / resample_freq
    t_start = np.ceil(t_instant[0] / dt) * dt
    t_end = np.floor(t_instant[-1] / dt) * dt
    t_resampled = np.arange(t_start, t_end + dt / 2, dt)
    rpm_resampled = np.interp(t_resampled, t_instant, instant_rpm)

    # Return with time relative to 0 (t_start as offset)
    return rpm_resampled, t_resampled - t_start, t_start


def compute_spectrogram_normalized(
    signal: np.ndarray,
    sample_rate: float,
    f_n: float,
    t_offset: float = 0.0,
    downsample_spec: int = 100,
    fft_seconds: float = 2.0,
    overlap_fraction: float = 0.9,
    mfft_factor: int = 4,
    f_max_ratio: float = 4.0,
) -> dict[str, np.ndarray]:
    """
    Compute spectrogram with frequency normalized by rotation frequency.

    Parameters
    ----------
    signal : np.ndarray
        Raw signal data at full sample rate.
    sample_rate : float
        Original sample rate [Hz].
    f_n : float
        Rotation frequency for normalization [Hz].
    t_offset : float
        Time offset to add to spectrogram time axis [s].
    downsample_spec : int
        Downsample factor for spectrogram computation.
    fft_seconds : float
        Window length for FFT [s].
    overlap_fraction : float
        Overlap between windows for FFT (0.9 = 90% overlap).
    mfft_factor : int
        Zero-padding factor (mfft = nperseg * mfft_factor).
    f_max_ratio : float
        Maximum f/f_n to include in output.

    Returns
    -------
    dict with keys:
        't': time centers [s]
        'f_ratio': frequency normalized by f_n [-]
        'Sxx_db': power spectral density [dB]
        'vmin', 'vmax': suggested color limits [dB]
    """
    # Downsample for efficiency
    fs_ds = sample_rate / downsample_spec
    data_spec = signal[::downsample_spec]

    # Spectrogram parameters
    nperseg = int(fs_ds * fft_seconds)
    hop = int(nperseg * (1 - overlap_fraction))
    mfft = nperseg * mfft_factor

    # Create ShortTimeFFT and compute
    win = hann(nperseg)
    SFT = ShortTimeFFT(win, hop, fs_ds, mfft=mfft, scale_to="psd")

    # Remove DC and compute spectrogram
    data_centered = data_spec - np.mean(data_spec)
    Sxx = SFT.spectrogram(data_centered)
    f = SFT.f
    t_spec = SFT.t(len(data_centered)) + t_offset

    # Normalize frequency by f_n and limit range
    f_ratio = f / f_n
    f_mask = f_ratio <= f_max_ratio
    f_ratio_plot = f_ratio[f_mask]
    Sxx_plot = Sxx[f_mask, :]

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx_plot + 1e-12)
    vmin, vmax = np.percentile(Sxx_db, [5, 99])

    return {
        "t": t_spec,
        "f_ratio": f_ratio_plot,
        "Sxx_db": Sxx_db,
        "vmin": vmin,
        "vmax": vmax,
    }


def plot_spectrogram_on_axis(
    ax: plt.Axes,
    spec_data: dict[str, np.ndarray],
    title: str = "",
    cmap: str = "hot",
) -> Any:
    """
    Plot spectrogram data on a matplotlib axis.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    spec_data : dict
        Output from compute_spectrogram_normalized().
    title : str
        Axis title.
    cmap : str
        Colormap name.

    Returns
    -------
    AxesImage from pcolormesh (for colorbar).
    """
    im = ax.pcolormesh(
        spec_data["t"],
        spec_data["f_ratio"],
        spec_data["Sxx_db"],
        shading="gouraud",
        cmap=cmap,
        vmin=spec_data["vmin"],
        vmax=spec_data["vmax"],
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("f/f_n [-]")
    ax.set_title(title)
    ax.set_ylim(0, spec_data["f_ratio"].max())

    return im


def add_gvo_overlay(
    ax: plt.Axes,
    t_gvo: np.ndarray,
    y_T: np.ndarray,
    color: str = "c",
) -> plt.Axes:
    """Add GVO trajectory overlay on secondary y-axis."""
    ax_gvo = ax.twinx()
    ax_gvo.plot(t_gvo, y_T, f"{color}-", lw=1.5, alpha=0.8)
    ax_gvo.set_ylabel("y_T [-]", color=color)
    ax_gvo.tick_params(axis="y", labelcolor=color)
    ax_gvo.set_ylim(0, 1)
    return ax_gvo


def align_signals_multi_rms(
    signal_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    sample_rate: float = 1.0,
    min_variance_ratio: float = 0.01,
    verbose: bool = True,
) -> tuple[float, dict[str, float], list[str]]:
    """
    Align signals using multiple signal pairs for robustness.

    Uses all signals that have sufficient variation. Normalizes each signal
    by its standard deviation so they contribute equally to the combined error.
    This handles cases where one or two signals are constant.

    Args:
        signal_pairs: List of (t_ref, signal_ref, t_target, signal_target, name) tuples.
                     Each tuple contains time and signal arrays for reference and target,
                     plus a name string for reporting.
        sample_rate: Resample rate for comparison [Hz] (default: 1 Hz)
        min_variance_ratio: Minimum variance ratio (signal_var / max_signal_var) to be
                           considered non-constant (default: 0.01 = 1%)
        verbose: Print which signals are skipped (default: True)

    Returns:
        Tuple of (best_offset, rms_dict, used_signals):
        - best_offset: Offset in seconds such that t_ref = t_target + offset
        - rms_dict: Dict mapping signal name to RMS error at best offset (original units)
        - used_signals: List of signal names that were used (non-constant)
    """
    dt = 1.0 / sample_rate

    # Resample all signals to common rate
    resampled = []  # (sig_ref, sig_target, name, variance)
    for t_ref, signal_ref, t_target, signal_target, name in signal_pairs:
        t_ref_rs = np.arange(t_ref[0], t_ref[-1], dt)
        t_target_rs = np.arange(t_target[0], t_target[-1], dt)
        sig_ref_rs = np.interp(t_ref_rs, t_ref, signal_ref)
        sig_target_rs = np.interp(t_target_rs, t_target, signal_target)
        # Combined variance of both signals
        var = (np.var(sig_ref_rs) + np.var(sig_target_rs)) / 2
        resampled.append((sig_ref_rs, sig_target_rs, name, var))

    # Filter out constant signals
    max_var = max(r[3] for r in resampled) if resampled else 1.0
    usable = []
    used_signals = []

    for sig_ref, sig_target, name, var in resampled:
        var_ratio = var / max_var if max_var > 0 else 0
        if var_ratio >= min_variance_ratio and var > 0:
            std = np.sqrt(var)
            # Store normalized signals and original std for denormalization
            usable.append((sig_ref / std, sig_target / std, name, std))
            used_signals.append(name)
        elif verbose:
            print(f"    Skipping {name}: constant (var_ratio={var_ratio:.4f})")

    if not usable:
        if verbose:
            print("    WARNING: No usable signals for alignment!")
        return 0.0, {}, []

    # All pairs should have same lengths after resampling from same time range
    n_ref = len(usable[0][0])
    n_target = len(usable[0][1])

    # Determine slide direction
    if n_target <= n_ref:
        sign = 1.0
        n_longer, n_shorter = n_ref, n_target
        get_pair = lambda u: (u[0], u[1])  # (longer=ref, shorter=target)
    else:
        sign = -1.0
        n_longer, n_shorter = n_target, n_ref
        get_pair = lambda u: (u[1], u[0])  # (longer=target, shorter=ref)

    # Slide and compute combined normalized RMS at each offset
    max_offset_samples = n_longer - n_shorter
    combined_rms = np.full(max_offset_samples + 1, np.inf)

    for offset in range(max_offset_samples + 1):
        total_sq = 0.0
        for u in usable:
            longer, shorter = get_pair(u)
            window = longer[offset : offset + n_shorter]
            if len(window) == n_shorter:
                total_sq += np.mean((window - shorter) ** 2)
        combined_rms[offset] = np.sqrt(total_sq / len(usable))

    # Find best offset
    best_idx = int(np.argmin(combined_rms))
    best_offset = sign * best_idx * dt

    # Compute individual RMS at best offset (in original units)
    rms_dict = {}
    for u in usable:
        longer, shorter = get_pair(u)
        name, std = u[2], u[3]
        window = longer[best_idx : best_idx + n_shorter]
        # Denormalize: multiply normalized RMS by std
        rms_dict[name] = np.sqrt(np.mean((window - shorter) ** 2)) * std

    return best_offset, rms_dict, used_signals


def align_signals_rms(
    t_ref: np.ndarray,
    signal_ref: np.ndarray,
    t_target: np.ndarray,
    signal_target: np.ndarray,
    sample_rate: float = 1.0,
) -> tuple[float, float]:
    """
    Align two signals by minimizing RMS error.

    Slides the shorter signal over the longer one and finds the offset
    that minimizes RMS error. Works regardless of which signal is longer.

    This is a convenience wrapper around align_signals_multi_rms for single
    signal pairs.

    Args:
        t_ref: Time array for reference signal [s]
        signal_ref: Reference signal values
        t_target: Time array for target signal [s]
        signal_target: Target signal values
        sample_rate: Resample rate for comparison [Hz] (default: 1 Hz)

    Returns:
        Tuple of (best_offset, rms_error):
        - best_offset: Offset in seconds such that t_ref = t_target + offset
        - rms_error: RMS error at best alignment
    """
    offset, rms_dict, _ = align_signals_multi_rms(
        [(t_ref, signal_ref, t_target, signal_target, "signal")],
        sample_rate=sample_rate,
        verbose=False,
    )
    return offset, rms_dict.get("signal", float("inf"))
