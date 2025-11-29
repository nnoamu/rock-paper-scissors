"""
MediaPipe + Random Forest training script.
FRISSÍTVE: Menti a feature-öket is más classifierek számára!

Használat:
    python scripts/train_mediapipe_rf.py
"""

import sys
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from core.data_object import DataObject

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor


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
        feature_vector = extractor.extract(data_obj)  # Fixed: was 'image'

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
    print("MediaPipe + Random Forest Training")
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

    extractor = MediaPipeHandExtractor()

    print("\n[1/5] Extracting training features...")
    X_train, y_train = extract_features_for_split(extractor, image_paths, train_indices)
    print(f"  ✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]}D features")

    print("\n[2/5] Extracting validation features...")
    X_val, y_val = extract_features_for_split(extractor, image_paths, val_indices)
    print(f"  ✓ Val: {X_val.shape[0]} samples")

    print("\n[3/5] Extracting test features...")
    X_test, y_test = extract_features_for_split(extractor, image_paths, test_indices)
    print(f"  ✓ Test: {X_test.shape[0]} samples")

    print("\n[4/5] Saving features for reuse...")
    features_dir = Path('data/mediapipe_features')
    features_dir.mkdir(parents=True, exist_ok=True)

    np.save(features_dir / 'X_train.npy', X_train)
    np.save(features_dir / 'y_train.npy', y_train)
    np.save(features_dir / 'X_val.npy', X_val)
    np.save(features_dir / 'y_val.npy', y_val)
    np.save(features_dir / 'X_test.npy', X_test)
    np.save(features_dir / 'y_test.npy', y_test)

    print(f"  ✓ Features saved to: {features_dir}")

    print("\n[5/5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
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

    model_path = Path('models/rf_mediapipe.pkl')
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Features available at: {features_dir}")
    print(f"\nNext steps:")
    print(f"  python scripts/train_mediapipe_dt.py  # Train faster Decision Tree")


if __name__ == '__main__':
    main()