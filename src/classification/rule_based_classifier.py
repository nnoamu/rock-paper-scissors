"""
Rule-based gesture classifier using finger openness.

No training required - works immediately based on geometric rules:
- ROCK: All fingers closed (low openness)
- PAPER: All fingers open (high openness)
- SCISSORS: Index and middle open, others closed

Works best with MediaPipeEnhancedExtractor which provides finger_openness in metadata.
Can also work with raw 63D landmarks by computing openness on-the-fly.
"""

import time
import numpy as np
from core.base_classifier import BaseClassifier
from core.feature_vector import FeatureVector
from core.classification_result import ClassificationResult, GestureClass
from core.data_object import DataObject
from typing import Dict, Optional, List


class RuleBasedGestureClassifier(BaseClassifier):
    """
    Rule-based classifier using finger openness values.

    Finger indices:
        0: Thumb
        1: Index
        2: Middle
        3: Ring
        4: Pinky

    Classification rules:
        ROCK: All fingers closed (openness < threshold)
        PAPER: All fingers open (openness > threshold)
        SCISSORS: Index + Middle open, others closed
    """

    def __init__(
        self,
        open_threshold: float = 0.65,
        closed_threshold: float = 0.45,
        confidence_base: float = 0.7
    ):
        """
        Initialize rule-based classifier.

        Args:
            open_threshold: Openness value above which finger is considered "open"
            closed_threshold: Openness value below which finger is considered "closed"
            confidence_base: Base confidence for classifications
        """
        super().__init__(name="Rule_Based_Gesture")
        self.open_threshold = open_threshold
        self.closed_threshold = closed_threshold
        self.confidence_base = confidence_base

    def _process(self, input: DataObject) -> ClassificationResult:
        start_time = time.perf_counter()

        features = input  # type: FeatureVector

        # Check if hand was detected
        if features.metadata and not features.metadata.get('hand_detected', True):
            return self._create_unknown_result(start_time, "No hand detected")

        # Get finger openness values
        finger_openness = self._get_finger_openness(features)

        if finger_openness is None:
            return self._create_unknown_result(start_time, "Could not extract finger openness")

        # Classify based on rules
        predicted_class, confidence, probabilities = self._classify_by_rules(finger_openness)

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000

        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=probabilities,
            processing_time_ms=processing_time,
            classifier_name=self.name,
            metadata={
                'finger_openness': finger_openness.tolist(),
                'open_threshold': self.open_threshold,
                'closed_threshold': self.closed_threshold
            }
        )

    def _get_finger_openness(self, features: FeatureVector) -> Optional[np.ndarray]:
        """
        Extract finger openness values from features.

        First tries metadata (from MediaPipeEnhancedExtractor),
        then tries named_features, otherwise returns None.
        """
        # Try metadata first (MediaPipeEnhancedExtractor provides this)
        if features.metadata and 'finger_openness' in features.metadata:
            return np.array(features.metadata['finger_openness'], dtype=np.float32)

        # Try named_features
        if features.named_features:
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            openness = []
            for name in finger_names:
                key = f'{name}_openness'
                if key in features.named_features:
                    openness.append(features.named_features[key])
                else:
                    return None
            return np.array(openness, dtype=np.float32)

        return None

    def _classify_by_rules(
        self,
        finger_openness: np.ndarray
    ) -> tuple[GestureClass, float, Dict[GestureClass, float]]:
        """
        Classify gesture based on finger openness rules.

        Returns:
            (predicted_class, confidence, class_probabilities)
        """
        thumb, index, middle, ring, pinky = finger_openness

        # Compute scores for each gesture
        rock_score = self._compute_rock_score(finger_openness)
        paper_score = self._compute_paper_score(finger_openness)
        scissors_score = self._compute_scissors_score(finger_openness)

        # Normalize to probabilities
        total = rock_score + paper_score + scissors_score + 1e-6
        probabilities = {
            GestureClass.ROCK: rock_score / total,
            GestureClass.PAPER: paper_score / total,
            GestureClass.SCISSORS: scissors_score / total,
            GestureClass.UNKNOWN: 0.0
        }

        # Find winner
        if rock_score >= paper_score and rock_score >= scissors_score:
            predicted = GestureClass.ROCK
            confidence = probabilities[GestureClass.ROCK]
        elif paper_score >= scissors_score:
            predicted = GestureClass.PAPER
            confidence = probabilities[GestureClass.PAPER]
        else:
            predicted = GestureClass.SCISSORS
            confidence = probabilities[GestureClass.SCISSORS]

        # Adjust confidence based on clarity of decision
        confidence = min(confidence * 1.2, 0.99)  # Boost but cap at 99%

        return predicted, confidence, probabilities

    def _compute_rock_score(self, openness: np.ndarray) -> float:
        """
        ROCK: All fingers closed.
        Score based on how closed all fingers are.
        """
        # All fingers should be below closed_threshold
        closed_scores = []
        for val in openness:
            if val < self.closed_threshold:
                closed_scores.append(1.0)
            elif val < self.open_threshold:
                # Partial score in the middle zone
                closed_scores.append(0.5)
            else:
                closed_scores.append(0.0)

        return np.mean(closed_scores)

    def _compute_paper_score(self, openness: np.ndarray) -> float:
        """
        PAPER: All fingers open.
        Score based on how open all fingers are.
        """
        open_scores = []
        for val in openness:
            if val > self.open_threshold:
                open_scores.append(1.0)
            elif val > self.closed_threshold:
                open_scores.append(0.5)
            else:
                open_scores.append(0.0)

        return np.mean(open_scores)

    def _compute_scissors_score(self, openness: np.ndarray) -> float:
        """
        SCISSORS: Index and middle open, thumb/ring/pinky closed.
        """
        thumb, index, middle, ring, pinky = openness

        scores = []

        # Index should be open
        if index > self.open_threshold:
            scores.append(1.0)
        elif index > self.closed_threshold:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Middle should be open
        if middle > self.open_threshold:
            scores.append(1.0)
        elif middle > self.closed_threshold:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Ring should be closed
        if ring < self.closed_threshold:
            scores.append(1.0)
        elif ring < self.open_threshold:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Pinky should be closed
        if pinky < self.closed_threshold:
            scores.append(1.0)
        elif pinky < self.open_threshold:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Thumb is flexible for scissors (can be open or closed)
        # Give partial score regardless
        scores.append(0.5)

        return np.mean(scores)

    def _create_unknown_result(
        self,
        start_time: float,
        reason: str
    ) -> ClassificationResult:
        """Create an UNKNOWN result with zero confidence."""
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000

        return ClassificationResult(
            predicted_class=GestureClass.UNKNOWN,
            confidence=0.0,
            class_probabilities={
                GestureClass.ROCK: 0.0,
                GestureClass.PAPER: 0.0,
                GestureClass.SCISSORS: 0.0,
                GestureClass.UNKNOWN: 1.0
            },
            processing_time_ms=processing_time,
            classifier_name=self.name,
            metadata={'reason': reason}
        )
