"""
Core modules package.
"""

from core.base_processor import PreprocessingModule
from core.base_feature_extractor import BaseFeatureExtractor
from core.base_classifier import BaseClassifier
from core.data_object import DataObject
from core.feature_vector import FeatureVector, FeatureType
from core.classification_result import ClassificationResult, GestureClass
from core.pipeline import ProcessingPipeline

__all__ = [
    'PreprocessingModule',
    'BaseFeatureExtractor',
    'BaseClassifier',
    'DataObject',
    'FeatureVector',
    'FeatureType',
    'ClassificationResult',
    'GestureClass',
    'ProcessingPipeline'
]