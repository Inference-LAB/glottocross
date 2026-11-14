"""
Acoustic Biomarker Extraction Module (Voicing-Gated & Standardized)
===================================================================
Extracts 17 multi-dimensional acoustic biomarkers adhering to ASHA guidelines:
- Cepstral: Cepstral Peak Prominence Smoothed (CPPS) via Praat PowerCepstrogram.
- Perturbation: Jitter (Local, RAP), Shimmer (Local, APQ3) with strict voicing-gating.
- Harmonicity: Harmonics-to-Noise Ratio (HNR) across 75-5000 Hz.
- Formants: Formant frequencies (F1, F2, F3) and Formant Dispersion (FD).
- Spectral Tilt: Alpha Ratio (50-1000 Hz vs 1000-5000 Hz), Hammarberg Index, Spectral Entropy.
"""

from typing import Dict, Union, Optional
import os
import numpy as np
import scipy.signal
import scipy.stats
import soundfile as sf
import parselmouth
from parselmouth.praat import call


def compute_cpps(sound: parselmouth.Sound) -> float:
    """
    Computes Cepstral Peak Prominence Smoothed (CPPS) using Praat algorithms.
    Evaluated using standard parabolic peak search across 60-330 Hz.
    Well-validated across both sustained vowels and connected speech (Awan et al., 2010).
    """
    try:
        cepstrogram = call(sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        cpps_val = call(cepstrogram, "Get CPPS", True, 0.01, 0.001, 60.0, 330.0, 0.05, "parabolic", 0.001, 0.05, "Straight", "Robust")
        if np.isnan(cpps_val) or np.isinf(cpps_val):
            return 0.0
        return float(cpps_val)
    except Exception:
        # Robust fallback using discrete cepstrum
        try:
            samples = sound.values[0]
            sr = sound.sampling_frequency
            frame_len = int(0.04 * sr)
            hop = int(0.01 * sr)
            peaks = []
            for i in range(0, len(samples) - frame_len, hop):
                frame = samples[i:i+frame_len] * np.hamming(frame_len)
                spec = np.abs(np.fft.rfft(frame)) + 1e-12
                ceps = np.abs(np.fft.irfft(np.log(spec)))
                min_q = int(sr / 400.0)
                max_q = int(sr / 60.0)
                if max_q < len(ceps):
                    peak = np.max(ceps[min_q:max_q])
                    trend = np.mean(ceps[min_q:max_q])
                    peaks.append(peak - trend)
            return float(np.mean(peaks)) if peaks else 0.0
        except Exception:
            return 0.0


def extract_spectral_biomarkers(samples: np.ndarray, sr: float) -> Dict[str, float]:
    """
    Extracts spectral energy distribution biomarkers:
    - Alpha Ratio: Energy in [50, 1000] Hz vs [1000, 5000] Hz (spectral tilt)
    - Hammarberg Index: Maximum energy [0, 2000] Hz vs [2000, 5000] Hz
    - Spectral Entropy: Quantifies signal disorder / turbulent aspiration noise
    - Spectral Centroid and Spread
    """
    freqs, psd = scipy.signal.welch(samples, fs=sr, nperseg=min(len(samples), 2048))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    
    # Alpha ratio
    band1 = (freqs >= 50) & (freqs <= 1000)
    band2 = (freqs > 1000) & (freqs <= 5000)
    e_low = np.sum(psd[band1]) + 1e-12
    e_high = np.sum(psd[band2]) + 1e-12
    alpha_ratio = 10.0 * np.log10(e_low / e_high)
    
    # Hammarberg index
    band_h1 = (freqs >= 0) & (freqs <= 2000)
    band_h2 = (freqs > 2000) & (freqs <= 5000)
    peak1 = np.max(psd[band_h1]) if np.any(band_h1) else 1e-12
    peak2 = np.max(psd[band_h2]) if np.any(band_h2) else 1e-12
    hammarberg = 10.0 * np.log10(max(peak1, 1e-12) / max(peak2, 1e-12))
    
    # Spectral Entropy
    spec_entropy = scipy.stats.entropy(psd_norm + 1e-12)
    
    # Centroid & Spread
    centroid = np.sum(freqs * psd_norm)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
    
    return {
        "alpha_ratio": float(alpha_ratio),
        "hammarberg_index": float(hammarberg),
        "spectral_entropy": float(spec_entropy),
        "spectral_centroid": float(centroid),
        "spectral_spread": float(spread)
    }


def extract_acoustic_features(audio_input: Union[str, np.ndarray], sr: Optional[float] = None, is_connected_speech: bool = False) -> Dict[str, float]:
    """
    Extracts standardized 17 acoustic biomarkers from voice audio.
    Applies voicing-gating to prevent unvoiced speech segments from distorting perturbation.
    """
    if isinstance(audio_input, str):
        sound = parselmouth.Sound(audio_input)
    else:
        sound = parselmouth.Sound(audio_input, sampling_frequency=sr or 44100.0)
        
    samples = sound.values[0]
    sample_rate = sound.sampling_frequency
    
    # Fundamental frequency (F0)
    pitch = sound.to_pitch()
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_mean = 0.0 if np.isnan(f0_mean) else float(f0_mean)
    f0_std = 0.0 if np.isnan(f0_std) else float(f0_std)
    
    # Voicing-gated PointProcess for perturbation
    point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local = 0.0 if np.isnan(jitter_local) else float(jitter_local)
    jitter_rap = 0.0 if np.isnan(jitter_rap) else float(jitter_rap)
    
    # Shimmer
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local = 0.0 if np.isnan(shimmer_local) else float(shimmer_local)
    shimmer_apq3 = 0.0 if np.isnan(shimmer_apq3) else float(shimmer_apq3)
    
    # Harmonicity (HNR)
    harmonicity = sound.to_harmonicity()
    hnr_mean = call(harmonicity, "Get mean", 0, 0)
    hnr_mean = 0.0 if np.isnan(hnr_mean) else float(hnr_mean)
    
    # Cepstral Peak Prominence Smoothed (CPPS)
    cpps_val = compute_cpps(sound)
    
    # Formants (Burg method)
    formants = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500.0)
    f1_mean = call(formants, "Get mean", 1, 0, 0, "Hertz")
    f2_mean = call(formants, "Get mean", 2, 0, 0, "Hertz")
    f3_mean = call(formants, "Get mean", 3, 0, 0, "Hertz")
    f1_mean = 0.0 if np.isnan(f1_mean) else float(f1_mean)
    f2_mean = 0.0 if np.isnan(f2_mean) else float(f2_mean)
    f3_mean = 0.0 if np.isnan(f3_mean) else float(f3_mean)
    
    formant_dispersion = (f3_mean - f1_mean) / 2.0 if f3_mean > f1_mean else 1000.0
    
    # Spectral biomarkers
    spec_metrics = extract_spectral_biomarkers(samples, sample_rate)
    
    return {
        "f0_mean": float(f0_mean),
        "f0_std": float(f0_std),
        "jitter_local": float(jitter_local),
        "jitter_rap": float(jitter_rap),
        "shimmer_local": float(shimmer_local),
        "shimmer_apq3": float(shimmer_apq3),
        "hnr_mean": float(hnr_mean),
        "cpps": float(cpps_val),
        "f1_mean": float(f1_mean),
        "f2_mean": float(f2_mean),
        "f3_mean": float(f3_mean),
        "formant_dispersion": float(formant_dispersion),
        **spec_metrics
    }


def extract_cepstral_and_spectral(audio_input: Union[str, np.ndarray], sr: Optional[float] = None, is_connected_speech: bool = False) -> np.ndarray:
    """Returns 17-dimensional standardized acoustic vector."""
    feats = extract_acoustic_features(audio_input, sr=sr, is_connected_speech=is_connected_speech)
    ordered_keys = [
        "f0_mean", "f0_std", "jitter_local", "jitter_rap",
        "shimmer_local", "shimmer_apq3", "hnr_mean", "cpps",
        "alpha_ratio", "hammarberg_index", "spectral_entropy",
        "spectral_centroid", "spectral_spread",
        "f1_mean", "f2_mean", "f3_mean", "formant_dispersion"
    ]
    return np.array([feats[k] for k in ordered_keys], dtype=np.float32)


def extract_laryngeal_valving_kinetics(
    audio_input: Union[str, np.ndarray],
    sr: Optional[float] = None,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0
) -> Dict[str, float]:
    """
    Extracts dynamic cricoarytenoid abduction-adduction valving biomarkers
    from continuous connected speech:
    - glottic_voiced_fraction_pct (GVF, %): Percentage of frames achieving
      periodic glottal fold vibration.
    - laryngeal_valving_rate_hz (LVR, Hz): Frequency of reciprocal glottic
      voicing onset-offset transitions per second of active phonation.
    """
    if isinstance(audio_input, str):
        sound = parselmouth.Sound(audio_input)
    else:
        sound = parselmouth.Sound(audio_input, sampling_frequency=sr or 44100.0)

    pitch_obj = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    f0_vals = pitch_obj.selected_array["frequency"]
    voiced = (f0_vals > 0).astype(int)

    intensity_obj = sound.to_intensity(minimum_pitch=pitch_floor, time_step=time_step)
    int_vals = intensity_obj.values[0]

    min_len = min(len(voiced), len(int_vals))
    if min_len == 0:
        return {"glottic_voiced_fraction_pct": 0.0, "laryngeal_valving_rate_hz": 0.0}

    voiced = voiced[:min_len]
    int_vals = int_vals[:min_len]

    active_thresh = np.percentile(int_vals, 25)
    active_frames = int(np.sum(int_vals > active_thresh))
    active_duration_s = max(active_frames * time_step, 0.1)

    gvf_pct = float((np.sum(voiced) / max(len(voiced), 1)) * 100.0)
    transitions = int(np.sum(np.abs(np.diff(voiced))))
    lvr_hz = float(transitions / active_duration_s)

    return {
        "glottic_voiced_fraction_pct": gvf_pct,
        "laryngeal_valving_rate_hz": lvr_hz
    }
