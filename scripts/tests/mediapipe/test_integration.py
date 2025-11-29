"""
Integration teszt: Teljes pipeline tesztelése (preprocess -> extract -> classify)
MediaPipe Hand Extractor + Random Forest Classifier használatával.

Használat:
    python scripts/test_integration.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from core.pipeline import ProcessingPipeline
from preprocessing.grayscale import GrayscaleConverter
from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor
from classification.random_forest_classifier import MediaPipeRFClassifier


def test_full_pipeline():
    """Teszteli a teljes pipeline-t egy test képpel."""
    print("=" * 60)
    print("Full Pipeline Integration Test")
    print("=" * 60)

    # 1. Pipeline setup
    print("\n[1/4] Setting up pipeline...")
    pipeline = ProcessingPipeline()

    preprocessor = GrayscaleConverter()
    extractor = MediaPipeHandExtractor()
    classifier = MediaPipeRFClassifier()

    try:
        classifier.load_model()
        print("  ✓ Model loaded")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return

    pipeline.add_preprocessing(preprocessor)
    pipeline.set_feature_extractor(extractor)
    pipeline.set_classifier(classifier)

    print("  ✓ Pipeline configured")

    # 2. Load test image - try multiple images until we find one with hand detected
    print("\n[2/4] Loading test image...")

    import json
    import random
    with open('data/splits/split_indices.json', 'r') as f:
        data = json.load(f)

    test_indices = data['splits']['test']
    image_paths = data['image_paths']

    # Try up to 20 random images to find one with detectable hand
    image = None
    test_image_path = None
    for _ in range(20):
        idx = random.choice(test_indices)
        test_image_path = Path('data/raw') / image_paths[idx]

        if not test_image_path.exists():
            continue

        temp_img = cv2.imread(str(test_image_path))
        if temp_img is None:
            continue

        # Quick check if hand is detected
        temp_features = extractor.extract(temp_img)
        if temp_features.metadata.get('hand_detected', False):
            image = temp_img
            print(f"  ✓ Loaded: {test_image_path.name} (hand detected)")
            print(f"    Shape: {image.shape}")
            break

    if image is None:
        print("  ⚠️  Warning: Could not find image with detectable hand in 20 tries")
        print("  Using first test image anyway...")
        test_image_path = Path('data/raw') / image_paths[test_indices[0]]
        image = cv2.imread(str(test_image_path))
        print(f"  ✓ Loaded: {test_image_path.name}")
        print(f"    Shape: {image.shape}")

    # 3. Run full pipeline
    print("\n[3/4] Running full pipeline...")
    try:
        # Step by step
        print("  → Preprocessing...")
        preprocessed = pipeline.preprocess_image(image)
        print(f"    ✓ Output shape: {preprocessed.shape}")

        print("  → Feature extraction...")
        features = pipeline.extract_features(preprocessed)
        print(f"    ✓ Features: {features.shape[0]}D")
        print(f"    ✓ Hand detected: {features.metadata.get('hand_detected', False)}")
        print(f"    ✓ Time: {features.extraction_time_ms:.2f}ms")

        print("  → Classification...")
        result = pipeline.classify(features)
        print(f"    ✓ Predicted: {result.predicted_class.value}")
        print(f"    ✓ Confidence: {result.confidence:.2%}")
        print(f"    ✓ Time: {result.processing_time_ms:.2f}ms")

    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test with process_full_pipeline() method
    print("\n[4/4] Testing process_full_pipeline() method...")
    try:
        preprocessed, features, final_result = pipeline.process_full_pipeline(image)
        print(f"  ✓ Preprocessing: {preprocessed.shape}")
        print(f"  ✓ Features: {features.shape[0]}D")
        print(f"  ✓ Result: {final_result.predicted_class.value}")
        print(f"  ✓ Confidence: {final_result.confidence:.2%}")
        print(f"  ✓ Classify time: {final_result.processing_time_ms:.2f}ms")

        print("\n  Probabilities:")
        for cls, prob in final_result.class_probabilities.items():
            bar = "█" * int(prob * 30)
            print(f"    {cls.value:8s}: {prob:6.2%} {bar}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ Integration test PASSED!")
    print("=" * 60)

    # Cleanup
    extractor.close()


if __name__ == '__main__':
    test_full_pipeline()
