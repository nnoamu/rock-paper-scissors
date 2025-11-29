"""
Egyszerű teszt script a MediaPipe + Random Forest modell tesztelésére.
Betölt néhány random képet a test setből és prediktál rájuk.

Használat:
    python scripts/test_mediapipe_model.py
    python scripts/test_mediapipe_model.py --num-samples 10
"""

import sys
import json
import random
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor
from classification.random_forest_classifier import MediaPipeRFClassifier
from core.classification_result import GestureClass


def load_test_images(num_samples=5):
    """Betölt random képeket a test setből."""
    split_file = 'data/splits/split_indices.json'
    with open(split_file, 'r') as f:
        data = json.load(f)

    test_indices = data['splits']['test']
    image_paths = data['image_paths']

    # Random válogatás
    selected_indices = random.sample(test_indices, min(num_samples, len(test_indices)))

    test_samples = []
    for idx in selected_indices:
        img_path = Path('data/raw') / image_paths[idx]

        # Parse ground truth label
        path_parts = image_paths[idx].replace('\\', '/').split('/')
        class_name = path_parts[0]

        label_map = {'rock': GestureClass.ROCK, 'paper': GestureClass.PAPER, 'scissors': GestureClass.SCISSORS}
        true_label = label_map.get(class_name, GestureClass.UNKNOWN)

        if img_path.exists():
            image = cv2.imread(str(img_path))
            if image is not None:
                test_samples.append({
                    'image': image,
                    'true_label': true_label,
                    'path': str(img_path)
                })

    return test_samples


def test_model(num_samples=5):
    """Teszteli a MediaPipe + RF modellt."""
    print("=" * 60)
    print("MediaPipe + Random Forest Model Test")
    print("=" * 60)

    # 1. Inicializálás
    print("\n[1/3] Initializing...")
    extractor = MediaPipeHandExtractor()
    classifier = MediaPipeRFClassifier()

    try:
        classifier.load_model()
        print("  ✓ Model loaded from: models/rf_mediapipe.pkl")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return

    # 2. Test képek betöltése
    print(f"\n[2/3] Loading {num_samples} test images...")
    test_samples = load_test_images(num_samples)
    print(f"  ✓ Loaded {len(test_samples)} images")

    # 3. Predikció és kiértékelés
    print("\n[3/3] Running predictions...\n")
    print("-" * 60)

    correct = 0
    total = 0

    for i, sample in enumerate(test_samples, 1):
        image = sample['image']
        true_label = sample['true_label']
        path = sample['path']

        # Feature extraction
        features = extractor.extract(image)

        # Classification
        result = classifier.classify(features)

        is_correct = result.predicted_class == true_label
        correct += int(is_correct)
        total += 1

        # Output
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}:")
        print(f"  Path:       {Path(path).name}")
        print(f"  True:       {true_label.value}")
        print(f"  Predicted:  {result.predicted_class.value}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Hand det.:  {features.metadata.get('hand_detected', False)}")
        print(f"  Extract:    {features.extraction_time_ms:.2f}ms")
        print(f"  Classify:   {result.processing_time_ms:.2f}ms")

        # Show probabilities
        print(f"  Probs:")
        for cls, prob in result.class_probabilities.items():
            if cls != GestureClass.UNKNOWN:
                bar = "█" * int(prob * 20)
                print(f"    {cls.value:8s}: {prob:.2%} {bar}")
        print()

    # Összegzés
    print("-" * 60)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    print("=" * 60)

    # Cleanup
    extractor.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test MediaPipe + RF model')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of test samples')
    args = parser.parse_args()

    test_model(args.num_samples)
