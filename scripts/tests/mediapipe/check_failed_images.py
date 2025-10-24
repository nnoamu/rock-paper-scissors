"""
Megnézi hogy milyen képeken nem talál kezet a MediaPipe.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor


def check_failed_detections(num_check=10):
    """Megnézi az első N képet és listázza a sikertelen detektálásokat."""

    # Load test indices
    with open('data/splits/split_indices.json', 'r') as f:
        data = json.load(f)

    test_indices = data['splits']['test'][:num_check]
    image_paths = data['image_paths']

    extractor = MediaPipeHandExtractor()

    print("=" * 60)
    print("MediaPipe Detection Check")
    print("=" * 60)

    failed_images = []
    success_images = []

    for idx in test_indices:
        img_path = Path('data/raw') / image_paths[idx]

        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        features = extractor.extract(image)
        detected = features.metadata.get('hand_detected', False)

        # Parse label
        path_parts = image_paths[idx].replace('\\', '/').split('/')
        class_name = path_parts[0]

        if detected:
            success_images.append((img_path.name, class_name, image.shape))
        else:
            failed_images.append((img_path.name, class_name, image.shape))

    print(f"\n✓ Detected hands: {len(success_images)}/{num_check}")
    print(f"✗ Failed to detect: {len(failed_images)}/{num_check}")

    if failed_images:
        print("\n" + "=" * 60)
        print("Failed detections:")
        print("=" * 60)
        for name, label, shape in failed_images:
            print(f"  • {name:25s} | {label:8s} | {shape}")

        print("\n💡 Suggestion:")
        print("  Check these images manually:")
        for name, label, shape in failed_images[:3]:
            print(f"    data/raw/{label}/{name}")

    if success_images:
        print("\n" + "=" * 60)
        print("Successful detections (sample):")
        print("=" * 60)
        for name, label, shape in success_images[:5]:
            print(f"  • {name:25s} | {label:8s} | {shape}")

    extractor.close()


if __name__ == '__main__':
    check_failed_detections(50)
