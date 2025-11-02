"""
Dataset letöltő orchestrator - regisztrálja és futtatja a downloader modulokat.

Használat:
    python scripts/download_dataset.py --list
    python scripts/download_dataset.py --source kaggle-drgfreeman
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from downloaders.kaggle_downloader import KaggleDownloader


DOWNLOADERS = {
    'kaggle-drgfreeman': lambda: KaggleDownloader('drgfreeman'),
    'kaggle-sanikamal': lambda: KaggleDownloader('sanikamal'),
}


def list_sources():
    print("="*60)
    print("📦 Available Dataset Sources")
    print("="*60)

    for source_key in DOWNLOADERS.keys():
        try:
            downloader = DOWNLOADERS[source_key]()
            print(f"\n{source_key}:")
            print(f"  {downloader.get_description()}")
        except Exception as e:
            print(f"\n{source_key}: [Error: {e}]")

    print("\n" + "="*60)


def download_dataset(source_key: str, output_dir: str = 'data/raw') -> bool:
    if source_key not in DOWNLOADERS:
        print(f"❌ Unknown source: {source_key}")
        print(f"\nAvailable sources:")
        for key in DOWNLOADERS.keys():
            print(f"  - {key}")
        print("\nUse --list for details")
        return False

    try:
        downloader = DOWNLOADERS[source_key]()

        print("="*60)
        print("🎮 Rock-Paper-Scissors Dataset Downloader")
        print("="*60)
        print(f"Source: {downloader.name}")
        print(f"Info:   {downloader.get_description()}")
        print("="*60)

        downloader.setup_directories()
        success = downloader.download(output_dir)

        if success:
            print("\n" + "="*60)
            print("✅ DOWNLOAD COMPLETE")
            print("="*60)
            print(f"\nDataset: {output_dir}/")
            print("\nNext steps:")
            print("  1. python scripts/dataset_info.py")
            print("  2. python scripts/create_splits.py")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("❌ DOWNLOAD FAILED")
            print("="*60)
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download Rock-Paper-Scissors dataset',
        epilog="""
Examples:
  python scripts/download_dataset.py --list
  python scripts/download_dataset.py --source kaggle-drgfreeman
  python scripts/download_dataset.py --source kaggle-sanikamal --output data/custom
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        choices=list(DOWNLOADERS.keys()),
        help='Dataset source'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available sources'
    )

    args = parser.parse_args()

    if args.list:
        list_sources()
        sys.exit(0)

    if not args.source:
        parser.print_help()
        print("\n💡 Use --list to see available sources")
        sys.exit(1)

    success = download_dataset(args.source, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()