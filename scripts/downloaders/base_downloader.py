"""
Absztrakt base osztály dataset downloader modulokhoz.

Használat:
    class MyDownloader(BaseDownloader):
        def download(self, target_dir: str) -> bool:
            # Implementation
            return True

        def get_description(self) -> str:
            return "Dataset description"
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDownloader(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, target_dir: str) -> bool:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    def setup_directories(self, base_dir: str = 'data'):
        base_path = Path(base_dir)

        directories = [
            base_path / 'raw' / 'rock',
            base_path / 'raw' / 'paper',
            base_path / 'raw' / 'scissors',
            base_path / 'splits',
            base_path / 'processed',
            Path('models')
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        (base_path / 'processed' / '.gitkeep').touch(exist_ok=True)
        (Path('models') / '.gitkeep').touch(exist_ok=True)

        print(f"✅ Directory structure created")

    def __str__(self) -> str:
        return f"{self.name}: {self.get_description()}"