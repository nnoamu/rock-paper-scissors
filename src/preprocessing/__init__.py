"""
Preprocessing modules package.
"""
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.channel_splitter import ChannelSplitter
from preprocessing.image_splitter import ImageSplitterModule

__all__ = ['GrayscaleConverter', 'ChannelSplitter', 'ImageSplitterModule']