"""
Dataset statisztikák és információk.

Használat:
    python scripts/dataset_info.py
    python scripts/dataset_info.py --data data/raw
"""

import argparse
from pathlib import Path
from collections import defaultdict


def analyze_dataset(dataset_dir: str = 'data/raw'):
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return

    print("=" * 60)
    print("📊 DATASET ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {dataset_path.absolute()}\n")

    classes = ['rock', 'paper', 'scissors']
    stats = {}
    total_images = 0

    for class_name in classes:
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            print(f"⚠️  Missing: {class_name}/")
            continue

        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.png')) + \
                 list(class_dir.glob('*.jpeg'))

        n_images = len(images)
        total_images += n_images

        formats = defaultdict(int)
        for img in images:
            formats[img.suffix] += 1

        stats[class_name] = {
            'count': n_images,
            'formats': dict(formats)
        }

    print("📂 Class Distribution:")
    print("-" * 60)
    for class_name in classes:
        if class_name in stats:
            count = stats[class_name]['count']
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"{class_name.capitalize():10s}: {count:5d} images ({percentage:5.1f}%)")
    print("-" * 60)
    print(f"{'TOTAL':10s}: {total_images:5d} images")

    print(f"\n📄 File Formats:")
    print("-" * 60)
    all_formats = defaultdict(int)
    for class_stats in stats.values():
        for fmt, count in class_stats['formats'].items():
            all_formats[fmt] += count

    for fmt, count in sorted(all_formats.items()):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{fmt:10s}: {count:5d} files ({percentage:5.1f}%)")

    print("\n" + "=" * 60)

    if total_images == 0:
        print("⚠️  No images found!")
        print("\nRun: python scripts/download_dataset.py")
    else:
        print("✅ Dataset OK")
        print("\nNext: python scripts/create_splits.py")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Rock-Paper-Scissors dataset'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/raw',
        help='Dataset directory (default: data/raw)'
    )

    args = parser.parse_args()
    analyze_dataset(args.data)


if __name__ == '__main__':
    main()