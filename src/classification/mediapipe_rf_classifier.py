"""
Random Forest classifier for MediaPipeHandExtractor (63D features).

Használat:
    classifier = MediaPipeRFClassifier()
    classifier.load_model()
    result = classifier.classify(features)
"""

from typing import Optional

from classification.base_mediapipe_rf_classifier import BaseMediaPipeRFClassifier


class MediaPipeRFClassifier(BaseMediaPipeRFClassifier):
    """
    Random Forest classifier for MediaPipeHandExtractor (63D features).

    Uses model: models/rf_mediapipe.pkl
    """

    EXPECTED_DIM = 63
    DEFAULT_MODEL_NAME = "rf_mediapipe.pkl"

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(name="Random_Forest_MediaPipe_63D", model_path=model_path)