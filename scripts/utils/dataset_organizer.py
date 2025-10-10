"""
Dataset struktúra normalizálása rock/paper/scissors mappákba.

Használat:
    from scripts.utils.dataset_organizer import organize_dataset
    organize_dataset(source_path, 'data/raw')
"""

import os
import shutil
from pathlib import Path


def organize_dataset(source_dir: Path, target_dir: str) -> bool:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Organizing dataset structure...")
    print(f"   Source: {source_dir}")
    print(f"   Target: {target_path}\n")

    class_dirs = {}

    for class_name in ['rock', 'paper', 'scissors']:
        found = False

        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)

            for dir_name in dirs:
                if dir_name.lower() == class_name.lower():
                    potential_dir = root_path / dir_name
                    images = _find_images(potential_dir)

                    if images:
                        class_dirs[class_name] = potential_dir
                        found = True
                        rel_path = potential_dir.relative_to(source_dir)
                        print(f"   ✓ {class_name:10s}: {rel_path} ({len(images)} images)")
                        break

            if found:
                break

    if len(class_dirs) != 3:
        print(f"\n⚠️  Found only {len(class_dirs)}/3 classes: {list(class_dirs.keys())}")
        print(f"\n🔍 Directory structure:")
        _print_directory_tree(source_dir, max_depth=3)
        return False

    print(f"\n📋 Copying to {target_dir}/")

    total_copied = 0
    for class_name, source_class_dir in class_dirs.items():
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(exist_ok=True)

        images = _find_images(source_class_dir)
        print(f"   {class_name:10s}: {len(images):4d} images")

        for i, img_path in enumerate(images):
            target_img = target_class_dir / f"{class_name}_{i:04d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, target_img)

        total_copied += len(images)

    print(f"\n✅ Dataset organized! Total: {total_copied} images")
    return True


def _find_images(directory: Path) -> list:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []

    for ext in extensions:
        images.extend(list(directory.glob(ext)))

    return images


def _print_directory_tree(path: Path, max_depth: int = 3, _current_depth: int = 0):
    if _current_depth >= max_depth:
        return

    indent = '  ' * _current_depth

    if _current_depth == 0:
        print(f"   {path.name}/")

    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for item in items[:10]:
            if item.is_dir():
                print(f"   {indent}  {item.name}/")
                _print_directory_tree(item, max_depth, _current_depth + 1)
            elif _current_depth < 2:
                print(f"   {indent}  {item.name}")

        if len(items) > 10:
            print(f"   {indent}  ... and {len(items) - 10} more")

    except PermissionError:
        print(f"   {indent}  [Permission Denied]")