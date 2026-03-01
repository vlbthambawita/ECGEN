"""ECG-specific plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def plot_ecg_leads(
    ecg: Tensor | np.ndarray,
    lead_names: list[str] = None,
    sampling_rate: int = 500,
    title: str = "ECG Signal",
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Plot all 12 ECG leads.
    
    Args:
        ecg: ECG signal of shape (num_leads, seq_length)
        lead_names: Names of leads (default: standard 12-lead names)
        sampling_rate: Sampling rate in Hz
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(ecg, Tensor):
        ecg = ecg.cpu().numpy()
    
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", 
                     "V1", "V2", "V3", "V4", "V5", "V6"]
    
    num_leads, seq_length = ecg.shape
    time = np.arange(seq_length) / sampling_rate
    
    fig, axes = plt.subplots(num_leads, 1, figsize=figsize, sharex=True)
    
    for i, (lead, ax) in enumerate(zip(ecg, axes)):
        ax.plot(time, lead, 'b-', linewidth=0.5)
        ax.set_ylabel(lead_names[i] if i < len(lead_names) else f"Lead {i+1}")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_ecg_comparison(
    real: Tensor | np.ndarray,
    generated: Tensor | np.ndarray,
    lead_idx: int = 0,
    lead_name: str = "Lead I",
    sampling_rate: int = 500,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Compare real and generated ECG signals for a single lead.
    
    Args:
        real: Real ECG signal
        generated: Generated ECG signal
        lead_idx: Index of lead to plot
        lead_name: Name of the lead
        sampling_rate: Sampling rate in Hz
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(real, Tensor):
        real = real.cpu().numpy()
    if isinstance(generated, Tensor):
        generated = generated.cpu().numpy()
    
    seq_length = real.shape[-1]
    time = np.arange(seq_length) / sampling_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    ax1.plot(time, real[lead_idx], 'b-', linewidth=0.8, label='Real')
    ax1.set_ylabel(f"{lead_name} (Real)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(time, generated[lead_idx], 'r-', linewidth=0.8, label='Generated')
    ax2.set_ylabel(f"{lead_name} (Generated)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    return fig


def plot_ecg_overlay(
    signals: list[Tensor | np.ndarray],
    labels: list[str],
    lead_idx: int = 0,
    lead_name: str = "Lead I",
    sampling_rate: int = 500,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Overlay multiple ECG signals for comparison.
    
    Args:
        signals: List of ECG signals
        labels: Labels for each signal
        lead_idx: Index of lead to plot
        lead_name: Name of the lead
        sampling_rate: Sampling rate in Hz
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for signal, label in zip(signals, labels):
        if isinstance(signal, Tensor):
            signal = signal.cpu().numpy()
        
        seq_length = signal.shape[-1]
        time = np.arange(seq_length) / sampling_rate
        ax.plot(time, signal[lead_idx], linewidth=0.8, label=label, alpha=0.7)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(lead_name)
    ax.set_title(f"ECG Comparison - {lead_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig
