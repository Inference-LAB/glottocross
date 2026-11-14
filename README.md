# GlottoCross (`glottocross`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**GlottoCross** is an open-source, biophysically grounded Python framework for **Otorhinolaryngology – Head and Neck Surgery (Laryngology, Phonosurgery, and Broncho-Esophagology)**. It quantifies within-speaker **Cross-Task Phonatory Dissociation ($\Xi_{\text{CTPD}}$)**, **Cyclic Cricoarytenoid Laryngeal Valving Kinetics ($\text{LVR}$, $\text{GVF}$)**, and **High-Speed Videolaryngoscopy ($HSV$) Glottal Area Waveform ($GAW$) Kinematics** to support pre-procedural differentiation of **senescent vocal fold atrophy (presbylaryngis)** from **Parkinsonian hypokinetic laryngopathy** prior to office-based injection medialization laryngoplasty (IML) or bilateral Type I thyroplasty.

---

## Clinical & Physiological Rationale

During routine laryngostroboscopy, patients are instructed to phonate a sustained vowel (`/a/` or `/i/`). Under static phonatory posture, patients with early neurogenic Parkinsonian laryngopathy frequently recruit compensatory supraglottic false vocal fold (ventricular) compression and elevated subglottic driving effort to force vocal fold approximation, temporarily masking glottic insufficiency and yielding deceptively elevated **Cepstral Peak Prominence Smoothed (CPPS)**.

When transitioning to continuous running speech—which requires rapid, cyclic alternation between posterior cricoarytenoid (PCA) abduction and lateral cricoarytenoid/thyroarytenoid (LCA/TA) re-adduction—this tonic compensation collapses. **GlottoCross** formalizes this dynamic laryngeal stress test by pairing tonic vowel baseline acoustics with dynamic connected-speech valving metrics.

---

## Key Features

1. **Standardized Glottic & Vocal Tract Acoustic Extraction (`glottocross.acoustics`)**
   - Extracts 17 ELS / AAO-HNS / ASHA-compliant acoustic biomarkers via Praat (`parselmouth`): fundamental frequency ($F_{0,\text{mean}}$, $F_{0,\text{std}}$), cycle-to-cycle perturbation (`jitter_local`, `jitter_rap`, `shimmer_local`, `shimmer_apq3`), harmonics-to-noise ratio (`hnr_mean`), smoothed cepstral peak prominence (`cpps`), vocal tract formants ($F_1, F_2, F_3$, formant dispersion), and spectral tilt/noise markers (`alpha_ratio`, `hammarberg_index`, `spectral_entropy`, `spectral_centroid`, `spectral_spread`).
2. **Dynamic Cricoarytenoid Valving Kinetics (`extract_laryngeal_valving_kinetics`)**
   - Directly quantifies **Cyclic Laryngeal Valving Rate ($\text{LVR}$, Hz)** (voicing onset-offset transitions per second of active phonation) and **Glottic Voiced Frame Fraction ($\text{GVF}$, %)** to measure cricoarytenoid abduction-adduction diadochokinesis.
3. **Within-Speaker Cross-Task Phonatory Dissociation (`glottocross.cross_task`)**
   - Computes 8 intra-subject differential breakdown metrics between sustained phonation and connected speech: $\Delta\text{CPPS}$, $R_{\text{CPPS}}$, $\Delta\text{HNR}$, Jitter Surge Ratio ($\text{JSR}$), $\Delta\alpha$, $\sigma_{\text{Formant}}$, $Z_{\text{CPPS}}$, and the composite index $\Xi_{\text{CTPD}}$.
4. **High-Speed Videolaryngoscopy ($HSV$) Glottal Kinematics (`glottocross.kinematics`)**
   - Computes Glottal Area Waveform ($GAW$), Glottal Gap Index ($GGI$), Glottal Asymmetry Index ($GAI$), Open Quotient ($OQ$), Speed Quotient ($SQ$), Closing Quotient ($ClQ$), and segmentation validation metrics (Dice Similarity Coefficient, IoU/Jaccard).
5. **Calibrated Diagnostic Classifiers with Clinical Abstention (`glottocross.models`)**
   - Provides `LightweightCrossTaskModel` (with probability calibration and equivocal-case abstention windows for in-person laryngostroboscopic referral) and `get_baseline_classifiers` across 6 machine learning families.

---

## Installation

### From PyPI

```bash
pip install glottocross
```

### From Source (Development)

```bash
git clone https://github.com/Inference-LAB/glottocross.git
cd glottocross
pip install -e .
```

---

## Quickstart Usage

### 1. Extract Acoustic Biomarkers, Laryngeal Valving Kinetics & Cross-Task Dissociation

```python
from glottocross import (
    extract_acoustic_features,
    extract_laryngeal_valving_kinetics,
    compute_cross_task_dissociation,
)

# 1. Extract 17 acoustic biomarkers from paired sustained vowel (/a/) and connected speech
vowel_feats = extract_acoustic_features("patient01_vowel.wav", is_connected_speech=False)
speech_feats = extract_acoustic_features("patient01_speech.wav", is_connected_speech=True)

# 2. Quantify within-speaker Cross-Task Phonatory Dissociation (CTPD)
ctpd = compute_cross_task_dissociation(vowel_feats, speech_feats)
print(f"Delta CPPS: {ctpd['delta_cpps']:.2f} dB | Xi_CTPD: {ctpd['xi_ctpd']:.3f}")

# 3. Extract dynamic cricoarytenoid abduction-adduction valving kinetics
valving = extract_laryngeal_valving_kinetics("patient01_speech.wav")
print(f"Laryngeal Valving Rate (LVR): {valving['laryngeal_valving_rate_hz']:.2f} Hz")
print(f"Glottic Voiced Fraction (GVF): {valving['glottic_voiced_fraction_pct']:.1f}%")
```

### 2. High-Speed Videolaryngoscopy ($HSV$) Glottal Kinematics

```python
import numpy as np
from glottocross import compute_glottal_area_waveform, extract_endoscopic_kinematics

# Example: sequence of grayscale endoscopic frames (N_frames, H, W)
frames = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(50)]

gaw = compute_glottal_area_waveform(frames)
kinematics = extract_endoscopic_kinematics(frames, fps=4000.0)
print("Endoscopic Kinematics:", kinematics)
```

### 3. Diagnostic Classification with Ambiguity Abstention

```python
import numpy as np
from glottocross import LightweightCrossTaskModel

# X: (N_subjects, 42) combined Vowel + Speech + CTPD feature matrix
X_train = np.random.randn(45, 42)
y_train = np.random.randint(0, 2, size=45)

model = LightweightCrossTaskModel(random_state=42, abstention_window=(0.35, 0.65))
model.fit(X_train, y_train)

probs = model.predict_proba(X_train)[:, 1]
preds, is_confident = model.predict_with_abstention(X_train)
```

---

## Running Unit Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use `glottocross` in your research, please cite:

> *Beyond Sustained Laryngostroboscopic Phonation: Cross-Task Glottic Valving Dissociation for Differentiating Presbylaryngis from Parkinsonian Hypokinetic Laryngopathy.* Division of Laryngology & Broncho-Esophagology / Inference LAB, 2026. Repository: https://github.com/Inference-LAB/glottocross

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
