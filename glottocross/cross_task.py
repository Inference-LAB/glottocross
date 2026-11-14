"""
Cross-Task Phonatory Dissociation Module (Reference-Normalized)
==============================================================
Formulates and computes the within-speaker Cross-Task Phonatory Dissociation Index (Xi_CTPD),
measuring the intra-subject breakdown in vocal stability when transitioning from static sustained
vowel phonation (/a/) to dynamic continuous connected speech (reading passage).

Central Clinical Question:
Can within-speaker cross-task phonatory acoustic biomarkers distinguish age-associated vocal
changes (presbyphonia) from Parkinsonian hypokinetic dysphonia?
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np


def compute_cross_task_dissociation(
    vowel_features: Dict[str, float],
    speech_features: Dict[str, float],
    ref_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Computes within-speaker cross-task phonatory dissociation metrics.
    
    Parameters
    ----------
    vowel_features : dict
        Acoustic features from sustained vowel (/a/).
    speech_features : dict
        Acoustic features from continuous connected speech.
    ref_stats : dict, optional
        Empirical mean and std of delta features in reference cohort (mu, sigma)
        for z-score normalization.
        
    Returns
    -------
    dict:
        xi_ctpd : Unified reference-normalized Cross-Task Phonatory Dissociation Index
        delta_cpps : CPPS drop from vowel to speech (dB)
        rel_cpps_collapse : Fractional CPPS collapse
        delta_hnr : HNR degradation (dB)
        jitter_surge_ratio : Ratio of continuous speech jitter to vowel jitter
        delta_alpha : Shift in spectral tilt (alpha ratio)
        formant_instability : Shift in formant dispersion
        z_cpps : Z-score of CPPS change relative to reference population
    """
    # CPPS drop
    cpps_v = max(vowel_features.get("cpps", 0.0), 0.1)
    cpps_s = max(speech_features.get("cpps", 0.0), 0.1)
    delta_cpps = cpps_v - cpps_s
    rel_cpps_collapse = delta_cpps / cpps_v
    
    # HNR drop
    hnr_v = vowel_features.get("hnr_mean", 10.0)
    hnr_s = speech_features.get("hnr_mean", 10.0)
    delta_hnr = hnr_v - hnr_s
    rel_hnr_drop = delta_hnr / max(abs(hnr_v), 2.0)
    
    # Jitter surge
    jit_v = max(vowel_features.get("jitter_local", 0.01), 0.001)
    jit_s = max(speech_features.get("jitter_local", 0.01), 0.001)
    jitter_surge_ratio = jit_s / jit_v
    log_jitter_ratio = float(np.log(max(jitter_surge_ratio, 0.01)))
    
    # Spectral Tilt Shift (Alpha ratio)
    alpha_v = vowel_features.get("alpha_ratio", 0.0)
    alpha_s = speech_features.get("alpha_ratio", 0.0)
    delta_alpha = alpha_v - alpha_s
    
    # Formant dispersion shift
    fd_v = vowel_features.get("formant_dispersion", 1000.0)
    fd_s = speech_features.get("formant_dispersion", 1000.0)
    formant_instability = abs(fd_v - fd_s) / max(fd_v, 500.0)
    
    # Reference Z-score normalization if provided
    if ref_stats is not None and "delta_cpps" in ref_stats:
        mu_c, s_c = ref_stats["delta_cpps"]
        z_cpps = (delta_cpps - mu_c) / (s_c + 1e-6)
    else:
        z_cpps = delta_cpps / 3.0  # default standardization
        
    # Standardized composite score (Xi_CTPD)
    xi_ctpd = float(
        0.40 * rel_cpps_collapse * 2.0 +
        0.25 * rel_hnr_drop +
        0.20 * max(0.0, log_jitter_ratio) +
        0.15 * (abs(delta_alpha) / 5.0)
    )
    
    return {
        "xi_ctpd": float(xi_ctpd),
        "delta_cpps": float(delta_cpps),
        "rel_cpps_collapse": float(rel_cpps_collapse),
        "delta_hnr": float(delta_hnr),
        "jitter_surge_ratio": float(jitter_surge_ratio),
        "delta_alpha": float(delta_alpha),
        "formant_instability": float(formant_instability),
        "z_cpps": float(z_cpps)
    }
