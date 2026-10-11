"""
Evaluation of generated ECGs: signal measurements and condition-adherence metrics.
"""

from ecgen.evaluation.measure import detect_r_peaks, heart_rate

__all__ = ["detect_r_peaks", "heart_rate"]
