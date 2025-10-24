"""
MediaPipe hand landmarks feature extractor - STABLE settings.
21 landmarks (x, y, z) → 63D feature vektor.

Settings (ezekkel lett trainelve az eredeti modell):
- static_image_mode=False (video tracking mode - jobban működik ezen a dataseten)
- model_complexity=0 (lite modell, gyors)
- min_detection_confidence=0.5
- min_tracking_confidence=0.5
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from core.base_feature_extractor import BaseFeatureExtractor
from core.feature_vector import FeatureVector, FeatureType


class MediaPipeHandExtractor(BaseFeatureExtractor):
    FEATURE_DIMENSION = 63

    def __init__(self):
        super().__init__(name="MediaPipe_Hand_Landmarks")
        self.mp_hands = mp.solutions.hands
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )

    def extract(self, preprocessed_image: np.ndarray) -> FeatureVector:
        start_time = time.perf_counter()

        if len(preprocessed_image.shape) == 2:
            image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
        elif preprocessed_image.shape[2] == 4:
            image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = preprocessed_image

        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            features = np.zeros(self.FEATURE_DIMENSION, dtype=np.float32)
            named_features = None
            metadata = {'hand_detected': False}
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = []

            for landmark in hand_landmarks.landmark:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

            features = np.array(landmarks_list, dtype=np.float32)
            named_features = None
            metadata = {'hand_detected': True, 'num_landmarks': 21}

        end_time = time.perf_counter()
        extraction_time = (end_time - start_time) * 1000

        return FeatureVector(
            feature_type=FeatureType.GEOMETRIC,
            extractor_name=self.name,
            extraction_time_ms=extraction_time,
            features=features,
            named_features=named_features,
            metadata=metadata
        )

    def get_feature_dimension(self) -> int:
        return self.FEATURE_DIMENSION

    def close(self):
        if self._hands:
            self._hands.close()
            self._hands = None

    def __del__(self):
        self.close()