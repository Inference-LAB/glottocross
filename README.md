# GlottoCross (`glottocross`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Manuscript Under Review](https://img.shields.io/badge/Manuscript-Under%20Review-orange.svg)](https://github.com/Inference-LAB/glottocross)

**GlottoCross** is an open-source, biophysically grounded Python framework for **Otorhinolaryngology – Head and Neck Surgery (Laryngology, Phonosurgery, and Broncho-Esophagology)**. It quantifies within-speaker **Cross-Task Phonatory Dissociation ($\Xi_{\text{CTPD}}$)**, **Cyclic Cricoarytenoid Laryngeal Valving Kinetics ($\text{LVR}$, $\text{GVF}$)**, and **High-Speed Videolaryngoscopy ($HSV$) Glottal Area Waveform ($GAW$) Kinematics** to support pre-procedural differentiation of **senescent vocal fold atrophy (presbylaryngis)** from **Parkinsonian hypokinetic laryngopathy**.

> **Publication Status:** The companion research manuscript introducing the `glottocross` framework (*"Beyond Sustained Laryngostroboscopic Phonation: Cross-Task Glottic Valving Dissociation for Differentiating Presbylaryngis from Parkinsonian Hypokinetic Laryngopathy"*) is currently **under peer review**.

- **Author:** Muhammad Khubaib Ahmad
- **Email:** [khubaib.ahmad@inference-lab.org](mailto:khubaib.ahmad@inference-lab.org)
- **Organization:** INFERENCE Lab
- **License:** MIT License

---

## Table of Contents

1. [Background](#background)
2. [What Problem Does GlottoCross Solve?](#what-problem-does-glottocross-solve)
3. [Architecture & Core Modules](#architecture--core-modules)
4. [Installation](#installation)
5. [Usage Guide & Examples](#usage-guide--examples)
   - [1. Acoustic Feature Extraction (Sustained Vowel vs. Connected Speech)](#1-acoustic-feature-extraction-sustained-vowel-vs-connected-speech)
   - [2. Dynamic Cricoarytenoid Laryngeal Valving Kinetics](#2-dynamic-cricoarytenoid-laryngeal-valving-kinetics)
   - [3. Cross-Task Phonatory Dissociation (CTPD)](#3-cross-task-phonatory-dissociation-ctpd)
   - [4. High-Speed Videolaryngoscopy (HSV) Glottal Kinematics](#4-high-speed-videolaryngoscopy-hsv-glottal-kinematics)
   - [5. Calibrated Diagnostic Triage & Ambiguity Abstention](#5-calibrated-diagnostic-triage--ambiguity-abstention)
6. [Limitations & Clinical Disclaimer](#limitations--clinical-disclaimer)
7. [Running Unit Tests](#running-unit-tests)
8. [Citation (Under Review)](#citation-under-review)
9. [License](#license)

---

## Background

In tertiary laryngology and broncho-esophagology clinics, older adults frequently present with complaints of progressive vocal fatigue, breathiness, reduced projection, and asthenia. Two primary etiologies dominate this geriatric clinical presentation:

1. **Presbylaryngis (Senescent Vocal Fold Atrophy):** Characterized by age-related elastin and collagen depletion within the superficial and deep layers of the lamina propria alongside thyroarytenoid (TA) sarcopenia. On laryngostroboscopy, presbylaryngis manifests as bilateral vocal fold bowing, prominence of the vocal processes, and a persistent spindle-shaped glottic chink. Crucially, intrinsic cricoarytenoid joint mobility and neurological motor drive remain preserved.
2. **Parkinsonian Hypokinetic Laryngopathy:** A neurogenic voice disorder driven by basal ganglia dopaminergic depletion, resulting in bradykinesia and lead-pipe rigidity of the intrinsic laryngeal musculature—specifically the thyroarytenoid (TA), lateral cricoarytenoid (LCA), and posterior cricoarytenoid (PCA) muscles—accompanied by impaired sensorimotor scaling of subglottic aerodynamic pressure.

Routine laryngostroboscopic and acoustic voice assessments (aligned with European Laryngological Society [ELS], AAO-HNS, and ASHA guidelines) rely heavily on **sustained vowel phonation** (`/a/` or `/i/`) because steady-state quasi-periodic vibration facilitates stroboscopic flash synchronization and perturbation tracking.

---

## What Problem Does GlottoCross Solve?

### 1. The Pre-Procedural Phonosurgical Triage Dilemma
While presbylaryngis and early Parkinsonian hypokinetic laryngopathy exhibit overlapping bowed glottic configurations and breathy dysphonia, their surgical and medical pathways diverge sharply:
- **Presbylaryngis** responds predictably to structural glottic narrowing via **office-based injection medialization laryngoplasty (IML)** (using calcium hydroxylapatite or hyaluronic acid fillers) or **bilateral Type I medialization thyroplasty**.
- **Parkinsonian Hypokinetic Laryngopathy**, by contrast, demonstrates suboptimal long-term functional outcomes following static vocal fold augmentation alone because volumetric filler does not resolve underlying cricoarytenoid neuromuscular rigidity or respiratory-phonatory hypokinesia. These patients require targeted neurological workup, laryngeal electromyography (LEMG), and intensive voice therapy (e.g., LSVT LOUD).

### 2. The Sustained-Vowel Masking Artifact
During a brief, isolated sustained vowel (`/a/`), patients with Parkinsonian laryngopathy can consciously recruit **compensatory supraglottic false vocal fold (ventricular) compression** and elevated subglottic expiratory drive to force vocal fold approximation. This tonic hyperfunctional compensation temporarily narrows the glottic chink, steepens glottal flow shut-off, and yields deceptively high **Cepstral Peak Prominence Smoothed (CPPS)** values—masking the underlying neurogenic deficit during routine sustained-vowel assessment.

### 3. The GlottoCross Solution: Dynamic Cross-Task Valving Stress Testing
Continuous running speech requires the larynx to act as a rapid biomechanical valve, executing 5 to 7 voicing onset-offset transitions per second via coordinated **PCA abduction** (for voiceless consonants) and **LCA/TA re-adduction** (for voiced segments). Under this dynamic neuromuscular load, Parkinsonian basal ganglia rigidity prevents rapid cricoarytenoid diadochokinesis, causing tonic compensation to collapse.

`glottocross` solves this diagnostic blind spot by:
- Extracting standardized acoustic markers separately from **tonic sustained vowel phonation** and **dynamic continuous speech**.
- Quantifying the intra-subject biomechanical breakdown via the **8-dimensional Cross-Task Phonatory Dissociation ($\Xi_{\text{CTPD}}$)** vector ($\Delta\text{CPPS}$, $R_{\text{CPPS}}$, $\Delta\text{HNR}$, $\text{JSR}$, $\Delta\alpha$, $\sigma_{\text{Formant}}$, $Z_{\text{CPPS}}$, $\Xi_{\text{CTPD}}$).
- Directly measuring **Cyclic Laryngeal Valving Rate ($\text{LVR}$, Hz)** and **Glottic Voiced Frame Fraction ($\text{GVF}$, %)** from continuous speech.
- Providing **High-Speed Videolaryngoscopy ($HSV$) Glottal Area Waveform ($GAW$) kinematics** ($GGI$, $GAI$, $OQ$, $SQ$, $ClQ$) and calibrated diagnostic classifiers with a **clinical abstention window** to flag equivocal cases for in-person laryngostroboscopy and LEMG.

---

## Architecture & Core Modules

| Module | Primary Functions / Classes | Description |
| :--- | :--- | :--- |
| `glottocross.acoustics` | `extract_acoustic_features`, `extract_cepstral_and_spectral`, `extract_laryngeal_valving_kinetics` | Extracts 17 ELS/ASHA-aligned acoustic biomarkers ($F_0$, jitter, shimmer, HNR, CPPS, $F_1\text{--}F_3$, formant dispersion, $\alpha$-ratio, Hammarberg index, spectral entropy/centroid/spread) plus dynamic cricoarytenoid valving kinetics ($\text{LVR}$, $\text{GVF}$). |
| `glottocross.cross_task` | `compute_cross_task_dissociation` | Computes the 8 within-speaker Cross-Task Phonatory Dissociation ($\Xi_{\text{CTPD}}$) biomarkers comparing sustained vowel baseline against connected speech. |
| `glottocross.kinematics` | `compute_glottal_area_waveform`, `extract_endoscopic_kinematics`, `compute_segmentation_metrics` | Quantifies endoscopic glottal area waveforms ($GAW$), Glottal Gap Index ($GGI$), Glottal Asymmetry Index ($GAI$), Open/Speed/Closing Quotients ($OQ, SQ, ClQ$), and segmentation Dice/IoU metrics. |
| `glottocross.models` | `LightweightCrossTaskModel`, `get_baseline_classifiers` | Sigmoid-calibrated ensemble classifier with built-in clinical abstention bounds $(p_{\text{low}}, p_{\text{high}})$ and standardized multi-family ML baselines. |

---

## Installation

### Install from PyPI (Recommended)

```bash
pip install glottocross
```

### Install from Source (Development & Testing)

```bash
git clone https://github.com/Inference-LAB/glottocross.git
cd glottocross
pip install -e ".[dev]"
```

### Verify Installation & Package Metadata

```bash
pip show glottocross
```

---

## Usage Guide & Examples

### 1. Acoustic Feature Extraction (Sustained Vowel vs. Connected Speech)

Extract all 17 standardized glottal source, perturbation, cepstral, and vocal tract spectral biomarkers from `.wav` recordings (or directly from 1D NumPy waveforms):

```python
from glottocross import extract_acoustic_features

# Extract from a sustained /a/ vowel recording (tonic phonatory posture)
vowel_features = extract_acoustic_features(
    "patient01_sustained_vowel.wav",
    is_connected_speech=False,
)

# Extract from a continuous reading passage / running speech (dynamic valving)
speech_features = extract_acoustic_features(
    "patient01_connected_speech.wav",
    is_connected_speech=True,
)

print(f"Sustained Vowel CPPS: {vowel_features['cpps']:.2f} dB")
print(f"Connected Speech CPPS: {speech_features['cpps']:.2f} dB")
print(f"Sustained Vowel HNR:  {vowel_features['hnr_mean']:.2f} dB")
print(f"Sustained Vowel Jitter (local): {vowel_features['jitter_local']:.3f}%")
```

**Returned Dictionary Keys (17 Biomarkers):**
- Fundamental Frequency: `f0_mean`, `f0_std`
- Cycle-to-Cycle Perturbation: `jitter_local`, `jitter_rap`, `shimmer_local`, `shimmer_apq3`
- Harmonic & Cepstral Periodicity: `hnr_mean`, `cpps`
- Supraglottic Formants: `f1_mean`, `f2_mean`, `f3_mean`, `formant_dispersion`
- Spectral Tilt & Glottal Noise: `alpha_ratio`, `hammarberg_index`, `spectral_entropy`, `spectral_centroid`, `spectral_spread`

---

### 2. Dynamic Cricoarytenoid Laryngeal Valving Kinetics

Quantify the rate of voicing onset-offset transitions (**Cyclic Laryngeal Valving Rate**, $\text{LVR}$) and the proportion of voiced frames (**Glottic Voiced Frame Fraction**, $\text{GVF}$) during continuous speech:

```python
from glottocross import extract_laryngeal_valving_kinetics

valving = extract_laryngeal_valving_kinetics(
    "patient01_connected_speech.wav",
    time_step=0.01,
    f0_min=75.0,
    f0_max=500.0,
)

print(f"Laryngeal Valving Rate (LVR): {valving['laryngeal_valving_rate_hz']:.2f} Hz")
print(f"Glottic Voiced Fraction (GVF): {valving['glottic_voiced_fraction_pct']:.2f}%")
print(f"Voicing Transitions: {valving['n_voicing_transitions']} across {valving['active_speech_duration_s']:.2f} s")
```

---

### 3. Cross-Task Phonatory Dissociation (CTPD)

Compute the 8-dimensional within-speaker dissociation vector comparing sustained vowel phonation against continuous speech:

```python
from glottocross import compute_cross_task_dissociation

ctpd = compute_cross_task_dissociation(vowel_features, speech_features)

for metric_name, value in ctpd.items():
    print(f"{metric_name:18s}: {value:.4f}")
```

**Output Biomarkers:**
- `delta_cpps`: $\Delta\text{CPPS} = \text{CPPS}_{\text{vowel}} - \text{CPPS}_{\text{speech}}$ (dB)
- `cpps_ratio`: $R_{\text{CPPS}} = \text{CPPS}_{\text{speech}} / \text{CPPS}_{\text{vowel}}$
- `delta_hnr`: $\Delta\text{HNR} = \text{HNR}_{\text{vowel}} - \text{HNR}_{\text{speech}}$ (dB)
- `jsr`: Jitter Surge Ratio ($\text{Jitter}_{\text{speech}} / \text{Jitter}_{\text{vowel}}$)
- `delta_alpha`: Spectral Tilt Shift ($\alpha_{\text{vowel}} - \alpha_{\text{speech}}$, dB)
- `sigma_formant`: Cross-task vocal tract formant stability index
- `z_cpps`: Normalized cepstral dissociation score
- `xi_ctpd`: Composite Cross-Task Phonatory Dissociation Index ($\Xi_{\text{CTPD}}$)

---

### 4. High-Speed Videolaryngoscopy (HSV) Glottal Kinematics

Extract Glottal Area Waveforms ($GAW$), kinematic quotients ($GGI$, $GAI$, $OQ$, $SQ$, $ClQ$), and segmentation accuracy metrics from endoscopic video frames or binary glottal masks:

```python
import numpy as np
from glottocross import (
    compute_glottal_area_waveform,
    extract_endoscopic_kinematics,
    compute_segmentation_metrics,
)

# Example: Sequence of 50 high-speed endoscopic frames (256x256 grayscale)
frames = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(50)]

# 1. Extract normalized Glottal Area Waveform GAW(t)
gaw = compute_glottal_area_waveform(frames, threshold=40)

# 2. Compute physiological glottal cycle kinematics at 4000 fps
kinematics = extract_endoscopic_kinematics(frames, fps=4000.0, threshold=40)
print(f"Glottal Gap Index (GGI):       {kinematics['GGI']:.3f}")
print(f"Glottal Asymmetry Index (GAI): {kinematics['GAI']:.3f}")
print(f"Open Quotient (OQ):            {kinematics['OQ']:.3f}")
print(f"Speed Quotient (SQ):           {kinematics['SQ']:.3f}")
print(f"Closing Quotient (ClQ):        {kinematics['ClQ']:.3f}")

# 3. Validate automated glottal segmentation against expert ground-truth mask
pred_mask = (frames[0] < 40).astype(np.uint8)
gt_mask = (frames[0] < 42).astype(np.uint8)
seg_metrics = compute_segmentation_metrics(pred_mask, gt_mask)
print(f"Dice: {seg_metrics['dice']:.4f} | IoU: {seg_metrics['iou']:.4f}")
```

---

### 5. Calibrated Diagnostic Triage & Ambiguity Abstention

Train and evaluate the calibrated ensemble classifier with an explicit **abstention window** to defer borderline acoustic profiles for in-person laryngostroboscopy and laryngeal electromyography (LEMG):

```python
import numpy as np
from glottocross import LightweightCrossTaskModel

# Combined feature matrix: 17 Vowel + 17 Speech + 8 CTPD = 42 features
rng = np.random.default_rng(42)
X_train = rng.normal(size=(45, 42))
y_train = rng.integers(0, 2, size=45)  # 0 = Presbylaryngis, 1 = Parkinsonian Laryngopathy

# Initialize calibrated model with a [0.35, 0.65] clinical abstention window
model = LightweightCrossTaskModel(random_state=42, abstention_window=(0.35, 0.65))
model.fit(X_train, y_train)

# Predict calibrated posterior probabilities
probabilities = model.predict_proba(X_train)[:, 1]

# Predict with clinical abstention (-1 indicates equivocal case referred for LEMG/stroboscopy)
predictions, is_confident = model.predict_with_abstention(X_train)
print(f"Confident triage rate: {is_confident.mean() * 100:.1f}%")
```

---

## Limitations & Clinical Disclaimer

Users and clinical researchers should account for the following methodological and translational limitations when applying `glottocross`:

1. **Complementary Biomarker Role:** Within-speaker Cross-Task Phonatory Dissociation ($\Xi_{\text{CTPD}}$) is designed as a **complementary cross-task representation** that enhances diagnostic discrimination and probability calibration when combined with baseline sustained-vowel and connected-speech acoustic features; it should not be used in isolation as a standalone single-feature diagnostic rule.
2. **Cohort Sample Size & Linguistic Generalizability:** The reference clinical validation was conducted on a retrospective geriatric cohort ($N = 45$; 21 Presbylaryngis, 24 Parkinsonian hypokinetic laryngopathy). Multi-center prospective validation across diverse languages, reading passages, and Unified Parkinson's Disease Rating Scale (UPDRS) severity strata is required prior to broad clinical deployment.
3. **Recording Hardware & Acoustic Environment Sensitivity:** Perturbation and spectral tilt markers ($\text{Jitter}$, $\text{Shimmer}$, $\alpha$-ratio, Hammarberg index) are sensitive to microphone-to-mouth distance, room reverberation, and background noise floor. Paired vowel and speech recordings from a given speaker must be acquired during the same session under identical hardware settings.
4. **Indirect Inference of Neuromuscular Kinetics:** While Cyclic Laryngeal Valving Rate ($\text{LVR}$) and $\Delta\text{CPPS}$ reflect functional cricoarytenoid abduction-adduction transition dynamics, acoustic signals do not directly measure motor unit action potentials or subglottic pressure. Definitive neuromuscular confirmation requires fine-wire **laryngeal electromyography (LEMG)** and direct **laryngostroboscopic** or **high-speed videolaryngoscopic ($HSV$)** visualization.
5. **Regulatory & Medical Disclaimer:** `glottocross` is provided strictly for **scientific research and clinical decision-support exploration**. It is **not** an FDA-cleared or CE-marked medical device and must never replace comprehensive examination by a board-certified otolaryngologist—head and neck surgeon or neurologist.

---

## Running Unit Tests

To run the full test suite covering acoustic extraction, laryngeal valving kinetics, cross-task dissociation, endoscopic kinematics, and calibrated classification:

```bash
pytest tests/ -v
```

---

## Citation (Under Review)

The research paper describing the clinical methodology, biophysics, and statistical validation of `glottocross` is currently **under peer review**. If you use this package in your work, please cite:

```bibtex
@article{ahmad2026glottocross,
  title   = {Beyond Sustained Laryngostroboscopic Phonation: Cross-Task Glottic Valving Dissociation for Differentiating Presbylaryngis from Parkinsonian Hypokinetic Laryngopathy},
  author  = {Ahmad, Muhammad Khubaib},
  journal = {Manuscript Under Review},
  year    = {2026},
  url     = {https://github.com/Inference-LAB/glottocross}
}
```

---

## License

This software is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for the full license text.

Copyright (c) 2026 **Muhammad Khubaib Ahmad** ([khubaib.ahmad@inference-lab.org](mailto:khubaib.ahmad@inference-lab.org)), INFERENCE Lab.
