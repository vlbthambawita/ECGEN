"""Signal quality metrics for ECG evaluation."""

import torch
import numpy as np
from torch import Tensor
from scipy import signal


def signal_to_noise_ratio(ecg: Tensor | np.ndarray) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Args:
        ecg: ECG signal
        
    Returns:
        SNR in dB
    """
    if isinstance(ecg, Tensor):
        ecg = ecg.cpu().numpy()
    
    signal_power = np.mean(ecg ** 2)
    noise_estimate = np.std(np.diff(ecg))
    noise_power = noise_estimate ** 2
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    return float(snr)


def baseline_wander(ecg: Tensor | np.ndarray, sampling_rate: int = 500) -> float:
    """
    Measure baseline wander in ECG signal.
    
    Args:
        ecg: ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Baseline wander metric
    """
    if isinstance(ecg, Tensor):
        ecg = ecg.cpu().numpy()
    
    # High-pass filter to remove baseline
    nyquist = sampling_rate / 2
    cutoff = 0.5 / nyquist
    b, a = signal.butter(4, cutoff, btype='high')
    filtered = signal.filtfilt(b, a, ecg)
    
    # Baseline is the difference
    baseline = ecg - filtered
    wander = np.std(baseline)
    
    return float(wander)


def powerline_interference(ecg: Tensor | np.ndarray, sampling_rate: int = 500) -> float:
    """
    Measure 50/60 Hz powerline interference.
    
    Args:
        ecg: ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Powerline interference metric
    """
    if isinstance(ecg, Tensor):
        ecg = ecg.cpu().numpy()
    
    # FFT
    fft = np.fft.fft(ecg)
    freqs = np.fft.fftfreq(len(ecg), 1/sampling_rate)
    power = np.abs(fft) ** 2
    
    # Power at 50 Hz and 60 Hz
    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_60 = np.argmin(np.abs(freqs - 60))
    
    powerline_power = power[idx_50] + power[idx_60]
    total_power = np.sum(power)
    
    interference = powerline_power / (total_power + 1e-8)
    
    return float(interference)


def calculate_quality_metrics(ecg: Tensor | np.ndarray, sampling_rate: int = 500) -> dict:
    """
    Calculate comprehensive signal quality metrics.
    
    Args:
        ecg: ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of quality metrics
    """
    return {
        "snr": signal_to_noise_ratio(ecg),
        "baseline_wander": baseline_wander(ecg, sampling_rate),
        "powerline_interference": powerline_interference(ecg, sampling_rate),
    }
