"""
Egyszerű dummy classifier teszteléshez.
Az első feature érték alapján küszöbözéssel dönt:
    < 0.33 → Rock
    0.33-0.66 → Paper
    > 0.66 → Scissors
"""

import time
from core.base_classifier import BaseClassifier
from core.feature_vector import FeatureVector
from core.classification_result import ClassificationResult, GestureClass
from core import DataObject
from typing import cast


class DummyClassifier(BaseClassifier):

    def __init__(self):
        super().__init__(name="Dummy_Classifier")

    def _process(self, input: DataObject) -> ClassificationResult:
        start_time = time.perf_counter()

        features=cast(FeatureVector, input)     #BaseClassifier default constraint miatt tudjuk
        first_feature = abs(features.features[0])

        if first_feature < 0.33:
            predicted = GestureClass.ROCK
            probs = {
                GestureClass.ROCK: 0.80,
                GestureClass.PAPER: 0.15,
                GestureClass.SCISSORS: 0.05,
                GestureClass.UNKNOWN: 0.0
            }
        elif first_feature < 0.66:
            predicted = GestureClass.PAPER
            probs = {
                GestureClass.ROCK: 0.10,
                GestureClass.PAPER: 0.75,
                GestureClass.SCISSORS: 0.15,
                GestureClass.UNKNOWN: 0.0
            }
        else:
            predicted = GestureClass.SCISSORS
            probs = {
                GestureClass.ROCK: 0.05,
                GestureClass.PAPER: 0.20,
                GestureClass.SCISSORS: 0.75,
                GestureClass.UNKNOWN: 0.0
            }

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000

        return ClassificationResult(
            predicted_class=predicted,
            confidence=probs[predicted],
            class_probabilities=probs,
            processing_time_ms=processing_time,
            classifier_name=self.name,
            metadata={
                'decision_feature': first_feature,
                'rule': 'threshold-based'
            }
        )