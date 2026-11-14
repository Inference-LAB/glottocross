"""
Endoscopic Glottal Kinematics & Segmentation Module (Study B)
============================================================
Provides independent benchmark evaluations on the BAGLS endoscopic benchmark:
1. Automated glottal lumen segmentation vs. expert ground-truth annotations (Dice, IoU).
2. Structural and geometric measurements from endoscopic high-speed videolaryngoscopy (HSV):
   - Glottal Area Waveform (GAW)
   - Glottal Gap Index (GGI)
   - Dynamic Glottal Area Asymmetry (GAI)
   - Open Quotient (OQ)
   - Phonovibratory Regularity Index (PRI)

NOTE: Image-derived glottal geometry reflects structural and morphological boundaries.
It does NOT directly measure aerodynamic airflow, subglottic pressure, or phonation threshold pressure.
Aerodynamic concepts are evaluated strictly as theoretical physiological mechanisms.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import scipy.signal


def compute_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    """
    Computes standard computer vision segmentation metrics against expert annotation:
    - Dice Similarity Coefficient (DSC)
    - Intersection-over-Union (IoU / Jaccard Index)
    - Precision and Recall
    """
    p_bin = (pred_mask > 0).astype(np.uint8)
    t_bin = (true_mask > 0).astype(np.uint8)
    
    intersection = float(np.sum(p_bin & t_bin))
    sum_p = float(np.sum(p_bin))
    sum_t = float(np.sum(t_bin))
    union = sum_p + sum_t - intersection
    
    dice = (2.0 * intersection) / (sum_p + sum_t + 1e-12) if (sum_p + sum_t) > 0 else 1.0
    iou = intersection / (union + 1e-12) if union > 0 else 1.0
    prec = intersection / (sum_p + 1e-12) if sum_p > 0 else 1.0
    rec = intersection / (sum_t + 1e-12) if sum_t > 0 else 1.0
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec)
    }


def compute_glottal_area_waveform(frames_or_masks: List[np.ndarray], is_mask: bool = False, threshold: Optional[float] = None) -> np.ndarray:
    """
    Computes the Glottal Area Waveform GAW(t) across a sequence of endoscopic frames or masks.
    """
    gaw = []
    for f in frames_or_masks:
        if f.ndim == 3:
            gray = np.mean(f, axis=2)
        else:
            gray = f
            
        if is_mask:
            area = float(np.sum(gray > 128))
        else:
            th = threshold if threshold is not None else np.percentile(gray, 15)
            area = float(np.sum(gray < th))
        gaw.append(area)
        
    gaw_arr = np.array(gaw, dtype=np.float64)
    if len(gaw_arr) >= 7:
        window_size = min(7, len(gaw_arr) if len(gaw_arr) % 2 != 0 else len(gaw_arr) - 1)
        gaw_arr = scipy.signal.savgol_filter(gaw_arr, window_length=window_size, polyorder=2)
    return np.maximum(gaw_arr, 0.0)


def compute_glottal_asymmetry(frames_or_masks: List[np.ndarray], midline_col: Optional[int] = None) -> float:
    """
    Computes Dynamic Glottal Area Asymmetry (GAI):
    GAI = Mean_t( |Area_left(t) - Area_right(t)| / (Area_left(t) + Area_right(t) + eps) )
    """
    asymmetries = []
    for f in frames_or_masks:
        if f.ndim == 3:
            gray = np.mean(f, axis=2)
        else:
            gray = f
            
        mid = midline_col if midline_col is not None else gray.shape[1] // 2
        left_region = gray[:, :mid]
        right_region = gray[:, mid:]
        
        th = np.percentile(gray, 15)
        area_left = float(np.sum(left_region < th))
        area_right = float(np.sum(right_region < th))
        
        denom = area_left + area_right + 1e-6
        if denom > 10.0:
            asymmetries.append(abs(area_left - area_right) / denom)
            
    if not asymmetries:
        return 0.0
    return float(np.mean(asymmetries))


def extract_endoscopic_kinematics(frames: List[np.ndarray], fps: float = 4000.0) -> Dict[str, float]:
    """
    Extracts structural laryngeal kinematics from high-speed videoendoscopy sequence:
    - glottal_gap_index (GGI): Min area / Max area (quantifies persistent open glottal area)
    - glottal_asymmetry_index (GAI): Dynamic left-right amplitude asymmetry
    - open_quotient (OQ): Proportion of cycle where glottis remains open
    - speed_quotient (SQ): Ratio of opening phase duration to closing phase duration
    - phonovibratory_regularity (PRI): Cycle-to-cycle vibratory consistency
    """
    gaw = compute_glottal_area_waveform(frames)
    
    if len(gaw) < 5 or np.max(gaw) <= 0.0:
        return {
            "glottal_gap_index": 0.0,
            "glottal_asymmetry_index": 0.0,
            "open_quotient": 0.5,
            "speed_quotient": 1.0,
            "phonovibratory_regularity": 0.0,
            "f0_hsv": 0.0
        }
        
    gaw_max = np.max(gaw)
    gaw_min = np.min(gaw)
    
    ggi = float(gaw_min / (gaw_max + 1e-12))
    gai = compute_glottal_asymmetry(frames)
    
    peaks, _ = scipy.signal.find_peaks(gaw, distance=max(2, int(fps / 500.0)), prominence=(gaw_max - gaw_min) * 0.15)
    valleys, _ = scipy.signal.find_peaks(-gaw, distance=max(2, int(fps / 500.0)))
    
    if len(peaks) >= 2:
        cycle_periods = np.diff(peaks) / fps
        mean_period = np.mean(cycle_periods)
        std_period = np.std(cycle_periods)
        f0_hsv = 1.0 / mean_period if mean_period > 0 else 0.0
        pri = max(0.0, min(1.0, float(1.0 - (std_period / (mean_period + 1e-12)))))
    else:
        f0_hsv = 0.0
        pri = 0.5
        
    threshold_open = gaw_min + 0.10 * (gaw_max - gaw_min)
    oq = float(np.mean(gaw > threshold_open))
    
    opening_durations = []
    closing_durations = []
    for p in peaks:
        pre_valleys = [v for v in valleys if v < p]
        post_valleys = [v for v in valleys if v > p]
        if pre_valleys and post_valleys:
            t_open = (p - pre_valleys[-1]) / fps
            t_close = (post_valleys[0] - p) / fps
            if t_close > 0:
                opening_durations.append(t_open)
                closing_durations.append(t_close)
                
    if opening_durations and closing_durations:
        sq = float(np.mean(opening_durations) / (np.mean(closing_durations) + 1e-12))
    else:
        sq = 1.05
        
    return {
        "glottal_gap_index": float(ggi),
        "glottal_asymmetry_index": float(gai),
        "open_quotient": float(oq),
        "speed_quotient": float(sq),
        "phonovibratory_regularity": float(pri),
        "f0_hsv": float(f0_hsv)
    }
