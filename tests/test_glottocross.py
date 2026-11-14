"""
Unit and Integration Tests for GlottoCross Package (Audited)
============================================================
"""

import os
import numpy as np
import pytest
from glottocross.acoustics import (
    extract_acoustic_features,
    extract_cepstral_and_spectral,
    extract_laryngeal_valving_kinetics,
)
from glottocross.kinematics import compute_glottal_area_waveform, extract_endoscopic_kinematics, compute_segmentation_metrics
from glottocross.cross_task import compute_cross_task_dissociation
from glottocross.models import LightweightCrossTaskModel, get_baseline_classifiers


def test_acoustic_extraction():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 450 * t)
    sig += 0.01 * np.random.normal(0, 1, len(t))
    
    feats = extract_acoustic_features(sig, sr=sr)
    assert "f0_mean" in feats
    assert "cpps" in feats
    assert "alpha_ratio" in feats
    assert feats["f0_mean"] > 140.0 and feats["f0_mean"] < 160.0
    
    vec = extract_cepstral_and_spectral(sig, sr=sr)
    assert len(vec) == 17
    assert not np.isnan(vec).any()


def test_laryngeal_valving_kinetics():
    sr = 44100
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    carrier = 0.5 * np.sin(2 * np.pi * 140 * t)
    # Alternate voiced (1) and unvoiced/silent (0) segments at ~4 Hz
    envelope = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(float)
    sig = carrier * envelope + 0.005 * np.random.normal(0, 1, len(t))
    valving = extract_laryngeal_valving_kinetics(sig, sr=sr)
    assert "glottic_voiced_fraction_pct" in valving
    assert "laryngeal_valving_rate_hz" in valving
    assert 20.0 <= valving["glottic_voiced_fraction_pct"] <= 80.0
    assert valving["laryngeal_valving_rate_hz"] > 0.0


def test_segmentation_metrics():
    pred = np.zeros((100, 100), dtype=np.uint8)
    true = np.zeros((100, 100), dtype=np.uint8)
    pred[20:60, 20:60] = 1
    true[20:60, 20:60] = 1
    res = compute_segmentation_metrics(pred, true)
    assert res["dice"] == pytest.approx(1.0, abs=1e-5)
    assert res["iou"] == pytest.approx(1.0, abs=1e-5)


def test_kinematics_extraction():
    frames = []
    for phase in np.linspace(0, 4 * np.pi, 20):
        frame = np.ones((100, 100), dtype=np.uint8) * 200
        radius_x = int(10 + 15 * (np.sin(phase) + 1.0))
        radius_y = 30
        y, x = np.ogrid[:100, :100]
        mask = ((x - 50)**2 / (radius_x + 1e-3)**2 + (y - 50)**2 / radius_y**2) <= 1
        frame[mask] = 20
        frames.append(frame)
        
    gaw = compute_glottal_area_waveform(frames)
    assert len(gaw) == 20
    assert np.max(gaw) > np.min(gaw)
    
    kin = extract_endoscopic_kinematics(frames, fps=1000.0)
    assert "glottal_gap_index" in kin
    assert "glottal_asymmetry_index" in kin
    assert "open_quotient" in kin
    assert 0.0 <= kin["glottal_gap_index"] <= 1.0


def test_cross_task_dissociation():
    vowel_feats = {
        "cpps": 12.5,
        "hnr_mean": 18.0,
        "jitter_local": 0.008,
        "alpha_ratio": -8.5,
        "formant_dispersion": 1100.0
    }
    mild_speech_feats = {
        "cpps": 10.5,
        "hnr_mean": 15.0,
        "jitter_local": 0.012,
        "alpha_ratio": -9.0,
        "formant_dispersion": 1090.0
    }
    path_speech_feats = {
        "cpps": 4.5,
        "hnr_mean": 7.0,
        "jitter_local": 0.038,
        "alpha_ratio": -15.0,
        "formant_dispersion": 980.0
    }
    
    mild_res = compute_cross_task_dissociation(vowel_feats, mild_speech_feats)
    path_res = compute_cross_task_dissociation(vowel_feats, path_speech_feats)
    
    assert path_res["xi_ctpd"] > mild_res["xi_ctpd"]
    assert path_res["rel_cpps_collapse"] > mild_res["rel_cpps_collapse"]


def test_lightweight_model():
    X = np.random.randn(30, 20)
    y = np.array([0]*15 + [1]*15)
    
    model = LightweightCrossTaskModel(random_state=42)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert probs.shape == (30, 2)
    preds, conf = model.predict_with_abstention(X)
    assert len(preds) == 30
    assert len(conf) == 30
    
    baselines = get_baseline_classifiers(random_state=42)
    assert len(baselines) == 6
