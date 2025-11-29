"""
Random Forest classifier for MediaPipeEnhancedExtractor (91D features).

Használat:
    classifier = MediaPipeEnhancedRFClassifier()
    classifier.load_model()
    result = classifier.classify(features)
"""

from typing import Optional

from classification.base_mediapipe_rf_classifier import BaseMediaPipeRFClassifier


class MediaPipeEnhancedRFClassifier(BaseMediaPipeRFClassifier):
    """
    Random Forest classifier for MediaPipeEnhancedExtractor (91D features).

    Uses model: models/rf_mediapipe_enhanced.pkl
    """

    EXPECTED_DIM = 91
    DEFAULT_MODEL_NAME = "rf_mediapipe_enhanced.pkl"

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(name="Random_Forest_MediaPipe_91D", model_path=model_path)