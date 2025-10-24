"""
Vizualizálja miért nem talál kezet bizonyos képeken.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor


def analyze_image(image_path: str):
    """Részletes elemzés egy képről."""

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"✗ Not found: {img_path}")
        return

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"✗ Failed to load: {img_path}")
        return

    print("=" * 60)
    print(f"Analyzing: {img_path.name}")
    print("=" * 60)

    # Image properties
    print(f"\nImage properties:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Value range: {image.min()}-{image.max()}")
    print(f"  Mean brightness: {image.mean():.1f}")

    # Check if it's grayscale disguised as RGB
    if len(image.shape) == 3:
        is_grayscale = np.allclose(image[:,:,0], image[:,:,1]) and np.allclose(image[:,:,1], image[:,:,2])
        print(f"  Effectively grayscale: {is_grayscale}")

    # Test MediaPipe
    extractor = MediaPipeHandExtractor()
    features = extractor.extract(image)

    print(f"\nMediaPipe results:")
    print(f"  Hand detected: {features.metadata.get('hand_detected', False)}")
    print(f"  Extraction time: {features.extraction_time_ms:.2f}ms")

    if not features.metadata.get('hand_detected', False):
        print("\n⚠️  FAILED TO DETECT HAND")
        print("\nPossible reasons:")
        print("  • Kéz túl kicsi vagy levágva")
        print("  • Szokatlan pózíció/szög")
        print("  • Rossz megvilágítás/háttér")
        print("  • CGI/rajzolt kéz (nem valódi fotó)")
        print("  • Ujjak összeérnek/takarják egymást")
    else:
        print("\n✓ Hand detected successfully")
        print(f"  Landmarks: {features.metadata.get('num_landmarks', 0)}")

    extractor.close()
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+', help='Image paths')
    args = parser.parse_args()

    for img in args.images:
        analyze_image(img)
