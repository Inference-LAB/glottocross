"""
GlottoCross: Cross-Task Glottic Valving & Acoustic-Endoscopic Kinematics Framework
==================================================================================
A lightweight, biophysically grounded computational framework for Otorhinolaryngology:
1. Acoustic voice biomarker extraction (CPPS, perturbation, spectral dynamics).
2. Dynamic cricoarytenoid abduction-adduction valving kinetics (LVR, GVF).
3. Endoscopic glottal kinematics quantification from laryngeal videostroboscopy/HSV.
4. Cross-task phonatory dissociation analysis (sustained vowel to continuous speech).
5. Pre-procedural differentiation of Presbylaryngis from Parkinsonian Laryngopathy.
"""

__version__ = "0.1.0"
__author__ = "Muhammad Khubaib Ahmad"
__email__ = "khubaib.ahmad@inference-lab.org"

from .acoustics import (
    extract_acoustic_features,
    extract_cepstral_and_spectral,
    extract_laryngeal_valving_kinetics,
)
from .kinematics import (
    compute_glottal_area_waveform,
    extract_endoscopic_kinematics,
    compute_segmentation_metrics,
)
from .cross_task import compute_cross_task_dissociation
from .models import LightweightCrossTaskModel, get_baseline_classifiers

__all__ = [
    "extract_acoustic_features",
    "extract_cepstral_and_spectral",
    "extract_laryngeal_valving_kinetics",
    "compute_glottal_area_waveform",
    "extract_endoscopic_kinematics",
    "compute_segmentation_metrics",
    "compute_cross_task_dissociation",
    "LightweightCrossTaskModel",
    "get_baseline_classifiers",
]
