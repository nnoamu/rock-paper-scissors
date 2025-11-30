"""
Debug script to check hand detection on specific images.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor


def test_image(image_path: str):
    """Test hand detection on a specific image."""
    print("=" * 60)
    print(f"Testing: {image_path}")
    print("=" * 60)

    extractor = MediaPipeHandExtractor()

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"✗ Image not found: {img_path}")
        return

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"✗ Failed to load image")
        return

    print(f"\nImage info:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Min/Max: {image.min()}/{image.max()}")

    # Extract features
    features = extractor.extract(image)

    print(f"\nMediaPipe Results:")
    print(f"  Hand detected: {features.metadata.get('hand_detected', False)}")
    print(f"  Extraction time: {features.extraction_time_ms:.2f}ms")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature min/max: {features.features.min():.3f}/{features.features.max():.3f}")
    print(f"  All zeros: {np.allclose(features.features, 0.0)}")

    # Check settings
    print(f"\nMediaPipe Settings:")
    print(f"  static_image_mode: False")
    print(f"  min_detection_confidence: 0.5")
    print(f"  min_tracking_confidence: 0.5")
    print(f"  model_complexity: 0")

    extractor.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image')
    args = parser.parse_args()

    test_image(args.image_path)
