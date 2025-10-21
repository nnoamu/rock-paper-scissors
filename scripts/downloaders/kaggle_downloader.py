"""
Kaggle dataset downloader kagglehub használatával.

Használat:
    downloader = KaggleDownloader('drgfreeman')
    downloader.download('data/raw')
"""

from pathlib import Path
from .base_downloader import BaseDownloader


class KaggleDownloader(BaseDownloader):

    AVAILABLE_DATASETS = {
        'drgfreeman': {
            'path': 'drgfreeman/rockpaperscissors',
            'images': 2188,
            'description': 'Változatos háttér és kéztartás'
        },
        'sanikamal': {
            'path': 'sanikamal/rock-paper-scissors-dataset',
            'images': 2520,
            'description': 'Nagyobb dataset, egyszerű háttér'
        }
    }

    def __init__(self, dataset_key: str = 'drgfreeman'):
        if dataset_key not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_key}. "
                f"Available: {list(self.AVAILABLE_DATASETS.keys())}"
            )

        self.dataset_key = dataset_key
        self.dataset_info = self.AVAILABLE_DATASETS[dataset_key]
        self.kaggle_path = self.dataset_info['path']

        super().__init__(name=f"Kaggle-{dataset_key}")

    def download(self, target_dir: str = 'data/raw') -> bool:
        print(f"\n📥 Downloading {self.kaggle_path}")
        print(f"   Expected: ~{self.dataset_info['images']} images")
        print("   First time: downloads to cache")
        print("   Subsequent runs: instant from cache\n")

        try:
            import kagglehub

            download_path = kagglehub.dataset_download(self.kaggle_path)
            print(f"✅ Download complete!")
            print(f"📂 Cached at: {download_path}\n")

            from utils.dataset_organizer import organize_dataset
            return organize_dataset(Path(download_path), target_dir)

        except ImportError:
            print("❌ kagglehub not installed!")
            print("\nInstall: pip install kagglehub")
            print("Configure: https://www.kaggle.com/docs/api")
            return False

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nCheck Kaggle authentication (kaggle.json)")
            return False

    def get_description(self) -> str:
        return (
            f"{self.dataset_info['images']} images - "
            f"{self.dataset_info['description']}"
        )

    @classmethod
    def list_available(cls) -> dict:
        return cls.AVAILABLE_DATASETS