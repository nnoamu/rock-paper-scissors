"""
Base class for MediaPipe Random Forest classifiers.
"""

import time
import pickle
from pathlib import Path
from abc import ABC
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from core.base_classifier import BaseClassifier
from core.feature_vector import FeatureVector
from core.classification_result import ClassificationResult, GestureClass
from core.data_object import DataObject
from typing import cast, Optional


class BaseMediaPipeRFClassifier(BaseClassifier, ABC):
    """Base class for MediaPipe Random Forest classifiers."""

    EXPECTED_DIM: int = 0
    DEFAULT_MODEL_NAME: str = ""

    def __init__(self, name: str, model_path: Optional[str] = None):
        super().__init__(name=name)
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[SklearnRF] = None
        self.label_map = {
            0: GestureClass.ROCK,
            1: GestureClass.PAPER,
            2: GestureClass.SCISSORS
        }
        self.is_trained = False

    def _get_default_model_path(self) -> Path:
        """Get default model path relative to this file's location."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / 'models' / self.DEFAULT_MODEL_NAME

    def load_model(self, model_path: Optional[str] = None):
        """Load the trained Random Forest model from disk."""
        if model_path:
            self.model_path = Path(model_path)
        elif self.model_path is None:
            self.model_path = self._get_default_model_path()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please train the model first using: python scripts/training/mediapipe/train_rf.py"
            )

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        if hasattr(self.model, 'n_features_in_'):
            if self.model.n_features_in_ != self.EXPECTED_DIM:
                raise ValueError(
                    f"Model dimension mismatch: expected {self.EXPECTED_DIM}D, "
                    f"but model was trained on {self.model.n_features_in_}D features."
                )

        self.is_trained = True
        print(f"Loaded RF model: {self.model_path} ({self.EXPECTED_DIM}D features)")

    def _process(self, input: DataObject) -> ClassificationResult:
        """Process feature vector and return classification result."""
        start_time = time.perf_counter()

        features = cast(FeatureVector, input)

        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        feature_dim = features.features.shape[0]
        if feature_dim != self.EXPECTED_DIM:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.EXPECTED_DIM}D, "
                f"got {feature_dim}D. Make sure you're using the correct extractor."
            )

        if np.allclose(features.features, 0.0):
            predicted = GestureClass.UNKNOWN
            probs = {
                GestureClass.ROCK: 0.0,
                GestureClass.PAPER: 0.0,
                GestureClass.SCISSORS: 0.0,
                GestureClass.UNKNOWN: 1.0
            }
            confidence = 0.0
        else:
            feature_vector = features.features.reshape(1, -1)
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]

            predicted = self.label_map.get(prediction, GestureClass.UNKNOWN)
            confidence = float(probabilities[prediction])

            probs = {
                GestureClass.ROCK: float(probabilities[0]),
                GestureClass.PAPER: float(probabilities[1]),
                GestureClass.SCISSORS: float(probabilities[2]),
                GestureClass.UNKNOWN: 0.0
            }

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000

        return ClassificationResult(
            predicted_class=predicted,
            confidence=confidence,
            class_probabilities=probs,
            processing_time_ms=processing_time,
            classifier_name=self.name,
            metadata={
                'model_path': str(self.model_path),
                'hand_detected': not np.allclose(features.features, 0.0)
            }
        )