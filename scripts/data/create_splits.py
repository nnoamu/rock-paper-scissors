"""
Train/validation/test split létrehozása stratified módszerrel.

Használat:
    python scripts/create_splits.py
    python scripts/create_splits.py --test-size 0.15 --val-size 0.15
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


CLASS_MAPPING = {'rock': 0, 'paper': 1, 'scissors': 2}


def count_images(dataset_dir: str) -> tuple:
    dataset_path = Path(dataset_dir)
    labels = []
    image_paths = []

    for class_name, label in CLASS_MAPPING.items():
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            print(f"⚠️  Missing: {class_dir}")
            continue

        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.png')) + \
                 list(class_dir.glob('*.jpeg'))

        for img in images:
            image_paths.append(str(img.relative_to(dataset_path)))
            labels.append(label)

    return image_paths, np.array(labels)


def create_splits(dataset_dir: str = 'data/raw',
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42,
                  output_path: str = 'data/splits/split_indices.json'):

    print("=" * 60)
    print("📊 Creating Dataset Splits")
    print("=" * 60)

    image_paths, labels = count_images(dataset_dir)
    n_samples = len(image_paths)

    if n_samples == 0:
        print("❌ No images found!")
        return

    print(f"\nTotal: {n_samples} images")
    for class_name, label in CLASS_MAPPING.items():
        count = np.sum(labels == label)
        print(f"  {class_name.capitalize():10s}: {count:4d} ({count / n_samples * 100:.1f}%)")

    indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=labels
    )

    val_ratio = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=labels[temp_idx]
    )

    splits = {
        'metadata': {
            'total_samples': int(n_samples),
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state,
            'dataset_dir': dataset_dir
        },
        'image_paths': image_paths,
        'labels': labels.tolist(),
        'splits': {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
    }

    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(train_idx):4d} ({len(train_idx) / n_samples * 100:.1f}%)")
    print(f"   Val:   {len(val_idx):4d} ({len(val_idx) / n_samples * 100:.1f}%)")
    print(f"   Test:  {len(test_idx):4d} ({len(test_idx) / n_samples * 100:.1f}%)")

    print("\n📊 Class distribution:")
    for split_name in ['train', 'val', 'test']:
        split_indices = splits['splits'][split_name]
        split_labels = labels[split_indices]
        print(f"\n  {split_name.capitalize()}:")
        for class_name, label in CLASS_MAPPING.items():
            count = np.sum(split_labels == label)
            print(f"    {class_name.capitalize():10s}: {count:4d}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n💾 Saved: {output_path}")
    print("=" * 60)
    print("✅ SPLITS CREATED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/raw',
        help='Dataset directory (default: data/raw)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test ratio (default: 0.2)'
    )

    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation ratio (default: 0.1)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/splits/split_indices.json',
        help='Output path'
    )

    args = parser.parse_args()

    create_splits(
        dataset_dir=args.data,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        output_path=args.output
    )


if __name__ == '__main__':
    main()