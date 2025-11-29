"""
MediaPipe Enhanced + Random Forest training script.
91D feature vector-t használ (normalized landmarks + derived features).

Hasznalat:
    python scripts/training/mediapipe/train_rf.py

Features (91D):
    - Normalized landmarks (60D): wrist-centered, scale-invariant
    - Finger angles (15D): 3 joints x 5 fingers
    - Fingertip-palm distances (5D)
    - Inter-finger distances (4D)
    - Finger openness (5D): 0=closed, 1=open
    - Hand orientation (2D): roll, pitch
"""

import sys
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from core.data_object import DataObject
from feature_extraction.mediapipe_enhanced_extractor import MediaPipeEnhancedExtractor


def load_split_indices(split_file='data/splits/split_indices.json'):
    with open(split_file, 'r') as f:
        data = json.load(f)
    return data


def extract_features_for_split(extractor, image_paths, indices, data_dir='data/raw'):
    features = []
    labels = []
    failed = 0

    print(f"Extracting features for {len(indices)} images...")

    for idx in tqdm(indices):
        img_path = Path(data_dir) / image_paths[idx]

        # Parse label from path (handles both / and \ separators)
        path_parts = image_paths[idx].replace('\\', '/').split('/')
        class_name = path_parts[0]

        label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        label = label_map.get(class_name, -1)

        if label == -1:
            continue

        if not img_path.exists():
            failed += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
            continue

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use DataObject as input
        data_obj = DataObject(image_rgb)
        feature_vector = extractor.extract(data_obj)

        if feature_vector.metadata.get('hand_detected', False):
            features.append(feature_vector.features)
            labels.append(label)
        else:
            failed += 1

    if failed > 0:
        print(f"  (!) {failed} images skipped (no hand detected or file error)")

    return np.array(features), np.array(labels)


def main():
    print("=" * 60)
    print("MediaPipe Enhanced (91D) + Random Forest Training")
    print("=" * 60)

    data = load_split_indices()
    image_paths = data['image_paths']
    train_indices = data['splits']['train']
    val_indices = data['splits']['val']
    test_indices = data['splits']['test']

    print(f"\nDataset info:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")

    extractor = MediaPipeEnhancedExtractor()

    print("\n[1/5] Extracting training features...")
    X_train, y_train = extract_features_for_split(extractor, image_paths, train_indices)
    print(f"  -> Train: {X_train.shape[0]} samples, {X_train.shape[1]}D features")

    print("\n[2/5] Extracting validation features...")
    X_val, y_val = extract_features_for_split(extractor, image_paths, val_indices)
    print(f"  -> Val: {X_val.shape[0]} samples")

    print("\n[3/5] Extracting test features...")
    X_test, y_test = extract_features_for_split(extractor, image_paths, test_indices)
    print(f"  -> Test: {X_test.shape[0]} samples")

    print("\n[4/5] Saving features for reuse...")
    features_dir = Path('data/mediapipe_enhanced_features')
    features_dir.mkdir(parents=True, exist_ok=True)

    np.save(features_dir / 'X_train.npy', X_train)
    np.save(features_dir / 'y_train.npy', y_train)
    np.save(features_dir / 'X_val.npy', X_val)
    np.save(features_dir / 'y_val.npy', y_val)
    np.save(features_dir / 'X_test.npy', X_test)
    np.save(features_dir / 'y_test.npy', y_test)

    # Save feature info
    feature_info = {
        'extractor': 'MediaPipeEnhancedExtractor',
        'dimension': 91,
        'features': {
            'normalized_landmarks': 60,
            'finger_angles': 15,
            'fingertip_palm_distances': 5,
            'inter_finger_distances': 4,
            'finger_openness': 5,
            'orientation': 2
        },
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test)
    }
    with open(features_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"  -> Features saved to: {features_dir}")

    print("\n[5/5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    test_pred = rf_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Val Accuracy:   {val_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"{'='*60}")

    # Detailed test results
    print("\nTest Set Classification Report:")
    class_names = ['rock', 'paper', 'scissors']
    print(classification_report(y_test, test_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"            Predicted")
    print(f"            Rock  Paper  Scissors")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:8s}  {row[0]:4d}  {row[1]:5d}  {row[2]:8d}")

    # Feature importance (top 10)
    print("\nTop 10 Most Important Features:")
    feature_names = []
    # Normalized landmarks (60)
    for i in range(1, 21):
        for coord in ['x', 'y', 'z']:
            feature_names.append(f'landmark_{i}_{coord}')
    # Finger angles (15)
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        for joint in ['mcp', 'pip', 'dip']:
            feature_names.append(f'{finger}_{joint}_angle')
    # Fingertip-palm distances (5)
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        feature_names.append(f'{finger}_palm_dist')
    # Inter-finger distances (4)
    feature_names.extend(['thumb_index_dist', 'index_middle_dist', 'middle_ring_dist', 'ring_pinky_dist'])
    # Finger openness (5)
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        feature_names.append(f'{finger}_openness')
    # Orientation (2)
    feature_names.extend(['hand_roll', 'hand_pitch'])

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Save model
    model_path = Path('models/rf_mediapipe_enhanced.pkl')
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"\n{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {features_dir}")
    print(f"{'='*60}")

    # Cleanup
    extractor.close()


if __name__ == '__main__':
    main()
