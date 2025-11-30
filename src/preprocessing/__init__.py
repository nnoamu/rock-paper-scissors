"""
Preprocessing modules package.
"""
from preprocessing.identity import IdentityPreprocessor
from preprocessing.downscale_int import DownscaleWithInterpolationPreprocessor
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.channel_splitter import ChannelSplitter
from preprocessing.image_splitter import ImageSplitterModule
from preprocessing.skin_color_segmenter.skin_color_segmenter import SkinColorSegmenterModule
from preprocessing.skin_color_segmenter.skin_color_segmenter_network import SkinColorSegmenterNetwork
from preprocessing.gaussian_blur import GaussianBlurModule
from preprocessing.edge_detector import EdgeDetectorModule
from preprocessing.threshold_fill import ThresholdFillModule
from preprocessing.hole_closing import HoleClosingModule
from preprocessing.edge_smoothing import EdgeSmoothingModule
from preprocessing.object_separator import ObjectSeparatorModule
from preprocessing.downscale import DownscaleModule

__all__ = [
    'IdentityPreprocessor',
    'DownscaleWithInterpolationPreprocessor',
    'GrayscaleConverter',
    'ChannelSplitter',
    'ImageSplitterModule',
    'SkinColorSegmenterModule',
    'SkinColorSegmenterNetwork',
    'GaussianBlurModule',
    'EdgeDetectorModule',
    'ThresholdFillModule',
    'HoleClosingModule',
    'EdgeSmoothingModule',
    'ObjectSeparatorModule',
    'DownscaleModule'
]