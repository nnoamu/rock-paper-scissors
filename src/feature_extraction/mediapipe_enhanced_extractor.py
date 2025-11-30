"""
Enhanced MediaPipe hand feature extractor.

Features:
- Normalized landmarks (wrist-centered, scale-invariant): 60D
- Finger angles (3 joints x 5 fingers): 15D
- Fingertip-palm distances: 5D
- Inter-finger distances: 4D
- Finger openness (0-1 per finger): 5D
- Hand orientation (roll, pitch): 2D

Total: 91D feature vector
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from core.base_feature_extractor import BaseFeatureExtractor
from core.feature_vector import FeatureVector, FeatureType
from core.data_object import DataObject
from typing import Dict, Optional


class MediaPipeEnhancedExtractor(BaseFeatureExtractor):
    """
    Enhanced MediaPipe feature extractor with normalized and derived features.

    Landmark indices (MediaPipe Hand):
        0: WRIST
        1-4: THUMB (CMC, MCP, IP, TIP)
        5-8: INDEX (MCP, PIP, DIP, TIP)
        9-12: MIDDLE (MCP, PIP, DIP, TIP)
        13-16: RING (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)
    """

    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Finger tip indices
    FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    # Finger MCP (base) indices for openness calculation
    FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    # Feature dimensions
    DIM_NORMALIZED_LANDMARKS = 60  # 20 landmarks (excluding wrist) x 3 coords
    DIM_FINGER_ANGLES = 15         # 3 angles x 5 fingers
    DIM_FINGERTIP_PALM = 5         # 5 fingertip-to-palm distances
    DIM_INTER_FINGER = 4           # 4 adjacent fingertip pairs
    DIM_FINGER_OPENNESS = 5        # 5 fingers
    DIM_ORIENTATION = 2            # roll, pitch

    FEATURE_DIMENSION = (DIM_NORMALIZED_LANDMARKS + DIM_FINGER_ANGLES +
                         DIM_FINGERTIP_PALM + DIM_INTER_FINGER +
                         DIM_FINGER_OPENNESS + DIM_ORIENTATION)  # 91

    def __init__(self):
        super().__init__(name="MediaPipe_Enhanced")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3,   # Lower threshold for better tracking
            model_complexity=1             # Higher complexity for better accuracy
        )

    def _process(self, input: DataObject) -> FeatureVector:
        start_time = time.perf_counter()

        preprocessed_image = input.data

        # Convert to RGB if needed
        if len(preprocessed_image.shape) == 2:
            image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
        elif preprocessed_image.shape[2] == 4:
            image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = preprocessed_image

        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            features = np.zeros(self.FEATURE_DIMENSION, dtype=np.float32)
            named_features = self._get_empty_named_features()
            metadata = {'hand_detected': False, 'landmarks': None}
        else:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract raw landmarks as numpy array (21 x 3)
            raw_landmarks = self._extract_raw_landmarks(hand_landmarks)

            # Compute all features
            normalized_landmarks = self._compute_normalized_landmarks(raw_landmarks)
            finger_angles = self._compute_finger_angles(raw_landmarks)
            fingertip_palm_distances = self._compute_fingertip_palm_distances(raw_landmarks)
            inter_finger_distances = self._compute_inter_finger_distances(raw_landmarks)
            finger_openness = self._compute_finger_openness(raw_landmarks)
            orientation = self._compute_orientation(raw_landmarks)

            # Concatenate all features
            features = np.concatenate([
                normalized_landmarks.flatten(),  # 60D
                finger_angles,                   # 15D
                fingertip_palm_distances,        # 5D
                inter_finger_distances,          # 4D
                finger_openness,                 # 5D
                orientation                      # 2D
            ]).astype(np.float32)

            named_features = self._build_named_features(
                finger_openness, finger_angles, orientation
            )

            metadata = {
                'hand_detected': True,
                'num_landmarks': 21,
                'landmarks': hand_landmarks,
                'finger_openness': finger_openness.tolist(),
                'orientation': orientation.tolist()
            }

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

    def _extract_raw_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract landmarks as (21, 3) numpy array."""
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ], dtype=np.float32)
        return landmarks

    def _compute_normalized_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks:
        - Center at wrist (landmark 0)
        - Scale by palm size (wrist to middle finger MCP distance)

        Returns (20, 3) array - excludes wrist since it's always (0,0,0).
        """
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]

        # Compute scale factor (palm size)
        palm_size = np.linalg.norm(middle_mcp - wrist)
        if palm_size < 1e-6:
            palm_size = 1.0  # Avoid division by zero

        # Normalize: center at wrist, scale by palm size
        normalized = (landmarks - wrist) / palm_size

        # Remove wrist (always 0,0,0)
        return normalized[1:]  # Shape: (20, 3)

    def _compute_finger_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute joint angles for each finger (3 angles per finger).

        For each finger, compute angles at:
        - MCP joint (knuckle)
        - PIP joint (middle joint)
        - DIP joint (end joint)

        Returns 15D array.
        """
        angles = []

        # Finger joint chains (from base to tip)
        finger_chains = [
            [self.WRIST, self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP],
            [self.WRIST, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP],
            [self.WRIST, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP],
            [self.WRIST, self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP],
            [self.WRIST, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP],
        ]

        for chain in finger_chains:
            # Compute 3 angles per finger (at joints 1, 2, 3 of the chain)
            for i in range(1, 4):
                angle = self._compute_angle(
                    landmarks[chain[i-1]],
                    landmarks[chain[i]],
                    landmarks[chain[i+1]]
                )
                angles.append(angle)

        return np.array(angles, dtype=np.float32)

    def _compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute angle at p2 formed by vectors p2->p1 and p2->p3. Returns angle in [0, 1] (normalized from [0, pi])."""
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5  # Default to middle value

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Normalize to [0, 1]
        return angle / np.pi

    def _compute_fingertip_palm_distances(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute distance from each fingertip to palm center.
        Palm center approximated as centroid of wrist and finger MCPs.

        Returns 5D array (normalized by palm size).
        """
        # Palm center: centroid of wrist and finger MCPs
        palm_points = [self.WRIST, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        palm_center = np.mean(landmarks[palm_points], axis=0)

        # Palm size for normalization
        palm_size = np.linalg.norm(landmarks[self.MIDDLE_MCP] - landmarks[self.WRIST])
        if palm_size < 1e-6:
            palm_size = 1.0

        distances = []
        for tip_idx in self.FINGERTIPS:
            dist = np.linalg.norm(landmarks[tip_idx] - palm_center) / palm_size
            distances.append(dist)

        return np.array(distances, dtype=np.float32)

    def _compute_inter_finger_distances(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute distances between adjacent fingertips.

        Returns 4D array: thumb-index, index-middle, middle-ring, ring-pinky.
        """
        palm_size = np.linalg.norm(landmarks[self.MIDDLE_MCP] - landmarks[self.WRIST])
        if palm_size < 1e-6:
            palm_size = 1.0

        distances = []
        for i in range(len(self.FINGERTIPS) - 1):
            dist = np.linalg.norm(
                landmarks[self.FINGERTIPS[i]] - landmarks[self.FINGERTIPS[i+1]]
            ) / palm_size
            distances.append(dist)

        return np.array(distances, dtype=np.float32)

    def _compute_finger_openness(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute openness (extension) of each finger as 0-1 value.

        0 = fully closed/bent (fist)
        1 = fully extended/open

        Method: Compare TIP distance from WRIST to MCP distance from WRIST.
        - Extended finger: TIP is farther from wrist than MCP
        - Closed finger: TIP curls back toward wrist, distance ratio is low

        Returns 5D array.
        """
        wrist = landmarks[self.WRIST]
        openness = []

        # Palm size for normalization
        palm_size = np.linalg.norm(landmarks[self.MIDDLE_MCP] - wrist)
        if palm_size < 1e-6:
            palm_size = 1.0

        # === THUMB ===
        # Thumb is tricky - in a fist, thumb is in front of other fingers
        # Use the angle at IP joint: closed thumb has sharp bend at IP
        thumb_mcp = landmarks[self.THUMB_MCP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_tip = landmarks[self.THUMB_TIP]

        # Vector from IP to MCP and from IP to TIP
        v1 = thumb_mcp - thumb_ip
        v2 = thumb_tip - thumb_ip

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            openness.append(0.5)
        else:
            # Angle at IP joint
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)  # 0 to pi

            # Straight thumb: angle ~ pi (180 degrees)
            # Bent thumb (fist): angle ~ pi/2 or less (90 degrees)
            # Map: angle pi -> 1.0 (open), angle pi/2 -> 0.0 (closed)
            thumb_open = np.clip((angle - np.pi/2) / (np.pi/2), 0.0, 1.0)
            openness.append(thumb_open)

        # === OTHER FINGERS ===
        finger_data = [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP),
        ]

        for mcp_idx, pip_idx, tip_idx in finger_data:
            mcp = landmarks[mcp_idx]
            pip = landmarks[pip_idx]
            tip = landmarks[tip_idx]

            # Distance from wrist to MCP (base reference)
            mcp_dist = np.linalg.norm(mcp - wrist)
            # Distance from wrist to TIP
            tip_dist = np.linalg.norm(tip - wrist)

            if mcp_dist < 1e-6:
                openness.append(0.5)
                continue

            # Ratio of TIP distance to MCP distance
            # Extended: ratio ~2.0-2.5 (TIP is far from wrist)
            # Closed: ratio ~0.8-1.2 (TIP curls back near wrist)
            ratio = tip_dist / mcp_dist

            # Map: ratio 1.0 -> 0 (TIP at same distance as MCP = closed)
            #      ratio 2.2 -> 1 (TIP much farther = fully extended)
            finger_open = np.clip((ratio - 1.0) / 1.2, 0.0, 1.0)
            openness.append(finger_open)

        return np.array(openness, dtype=np.float32)

    def _compute_orientation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute hand orientation (roll and pitch).

        Based on palm normal vector computed from wrist and finger MCPs.

        Returns 2D array: [roll, pitch] normalized to [-1, 1].
        """
        # Define palm plane using wrist, index MCP, and pinky MCP
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        # Two vectors in palm plane
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Palm normal (cross product)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        normal = normal / norm

        # Roll: rotation around forward axis (z component of normal)
        # Pitch: rotation around side axis (y component of normal)
        roll = np.arctan2(normal[0], normal[2]) / np.pi  # Normalized to [-1, 1]
        pitch = np.arcsin(np.clip(normal[1], -1.0, 1.0)) / (np.pi / 2)  # Normalized to [-1, 1]

        return np.array([roll, pitch], dtype=np.float32)

    def _build_named_features(
        self,
        finger_openness: np.ndarray,
        finger_angles: np.ndarray,
        orientation: np.ndarray
    ) -> Dict[str, float]:
        """Build dictionary of named features for interpretability."""
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        named = {}

        # Finger openness
        for i, name in enumerate(finger_names):
            named[f'{name}_openness'] = float(finger_openness[i])

        # Finger angles (3 per finger)
        joint_names = ['mcp', 'pip', 'dip']
        for i, fname in enumerate(finger_names):
            for j, jname in enumerate(joint_names):
                named[f'{fname}_{jname}_angle'] = float(finger_angles[i * 3 + j])

        # Orientation
        named['hand_roll'] = float(orientation[0])
        named['hand_pitch'] = float(orientation[1])

        return named

    def _get_empty_named_features(self) -> Dict[str, float]:
        """Return named features dict with zeros when no hand detected."""
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        joint_names = ['mcp', 'pip', 'dip']

        named = {}
        for fname in finger_names:
            named[f'{fname}_openness'] = 0.0
            for jname in joint_names:
                named[f'{fname}_{jname}_angle'] = 0.0

        named['hand_roll'] = 0.0
        named['hand_pitch'] = 0.0

        return named

    def get_feature_dimension(self) -> int:
        return self.FEATURE_DIMENSION

    def visualize(self, image: np.ndarray, features: FeatureVector) -> np.ndarray:
        """Draw hand landmarks and feature info on the image."""
        annotated_image = image.copy()

        if len(annotated_image.shape) == 2:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)

        if features.metadata and features.metadata.get('hand_detected', False):
            landmarks = features.metadata.get('landmarks')
            if landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            # Draw finger openness values
            finger_openness = features.metadata.get('finger_openness', [])
            if finger_openness:
                finger_names = ['T', 'I', 'M', 'R', 'P']
                y_pos = 30
                for i, (name, val) in enumerate(zip(finger_names, finger_openness)):
                    text = f"{name}: {val:.2f}"
                    cv2.putText(annotated_image, text, (10, y_pos + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_image

    def close(self):
        if self._hands:
            self._hands.close()
            self._hands = None

    def __del__(self):
        self.close()
