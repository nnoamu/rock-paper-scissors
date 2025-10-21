"""
Dataset downloader modulok.
"""

from .base_downloader import BaseDownloader
from .kaggle_downloader import KaggleDownloader

__all__ = [
    'BaseDownloader',
    'KaggleDownloader',
]