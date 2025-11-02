"""
Random Forest classifier MediaPipe hand landmark feature-ökhöz.
Betölti a betanított modellt és osztályoz vele.

Példa használat:
    classifier = MediaPipeRFClassifier('models/rf_mediapipe.pkl')
    result = classifier.classify(features)
"""

import time
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from core.base_classifier import BaseClassifier
from core.feature_vector import FeatureVector
from core.classification_result import ClassificationResult, GestureClass
from core.data_object import DataObject
from typing import cast


class MediaPipeRFClassifier(BaseClassifier):

    def __init__(self, model_path: str = 'models/rf_mediapipe.pkl'):
        super().__init__(name="Random_Forest_MediaPipe")
        self.model_path = Path(model_path)
        self.model: SklearnRF = None
        self.label_map = {
            0: GestureClass.ROCK,
            1: GestureClass.PAPER,
            2: GestureClass.SCISSORS
        }
        self.is_trained = False

    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please train the model first using: python scripts/train_mediapipe_rf.py"
            )

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True

    def _process(self, input: DataObject) -> ClassificationResult:
        start_time = time.perf_counter()

        # BaseClassifier constraint ensures this is a FeatureVector
        features = cast(FeatureVector, input)

        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if features.shape[0] != 63:
            raise ValueError(
                f"Expected 63D MediaPipe features, got {features.shape[0]}D. "
                f"Make sure you're using MediaPipeHandExtractor."
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