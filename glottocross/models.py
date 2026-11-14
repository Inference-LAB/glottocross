"""
Diagnostic Classification & Baseline Modeling Module (Study A)
==============================================================
Provides classifiers, baselines, and ablation models for differentiating
Age-associated vocal aging / presbyphonia from Parkinsonian hypokinetic dysphonia:
- Multi-Layer Perceptron (MLP)
- Regularized Logistic Regression (L2)
- Linear Support Vector Machine (Linear SVM)
- Radial Basis Function SVM (RBF SVM)
- Random Forest Classifier
- Histogram-based Gradient Boosting (HistGradientBoosting)

Features selective classification / abstention handling for equivocal predictions.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, brier_score_loss


class LightweightCrossTaskModel:
    """
    Lightweight neural classifier for cross-task voice diagnosis.
    Includes abstention threshold for ambiguous cases.
    """
    def __init__(self, random_state: int = 42, abstention_window: Tuple[float, float] = (0.35, 0.65)):
        self.random_state = random_state
        self.abstention_window = abstention_window
        self.scaler = StandardScaler()
        
        self.clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=16,
            learning_rate_init=0.005,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train classifier strictly on training fold."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities [P(Presbyphonia), P(Parkinsonian)]."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns standard binary prediction."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
        
    def predict_with_abstention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, is_confident_mask).
        If probability is within abstention_window, marks as non-confident (requires in-person exam).
        """
        probs = self.predict_proba(X)[:, 1]
        low, high = self.abstention_window
        confident_mask = (probs < low) | (probs > high)
        preds = (probs >= 0.5).astype(int)
        return preds, confident_mask


def get_baseline_classifiers(random_state: int = 42) -> Dict[str, object]:
    """Returns suite of standard clinical machine learning baselines."""
    return {
        "Logistic_Regression": LogisticRegression(C=1.0, max_iter=500, random_state=random_state),
        "Linear_SVM": SVC(kernel="linear", probability=True, C=1.0, random_state=random_state),
        "RBF_SVM": SVC(kernel="rbf", probability=True, C=1.0, random_state=random_state),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state),
        "Gradient_Boosting": HistGradientBoostingClassifier(max_iter=50, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=random_state),
        "Lightweight_MLP": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=500, random_state=random_state)
    }
