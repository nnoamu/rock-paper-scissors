"""
Teljes 3-fázisú feldolgozó pipeline.

Fázisok:
1. Preprocessing - Képjavítás
2. Feature Extraction - Jellemzők kinyerése
3. Classification - Döntéshozatal

Példa használat:
    pipeline = ProcessingPipeline()
    pipeline.add_preprocessing(GrayscaleConverter())
    pipeline.set_feature_extractor(MyExtractor())
    pipeline.set_classifier(MyClassifier())

    preprocessed, features, result, annotated = pipeline.process_full_pipeline(image)
"""

from typing import List, Optional
import numpy as np

from .base_processor import PreprocessingModule
from .base_feature_extractor import BaseFeatureExtractor
from .base_classifier import BaseClassifier
from .feature_vector import FeatureVector
from .classification_result import ClassificationResult
from .data_object import DataObject

class ProcessingPipeline:

    def __init__(self):
        self.preprocessing_modules: List[PreprocessingModule] = []
        self.feature_extractor: Optional[BaseFeatureExtractor] = None
        self.classifier: Optional[BaseClassifier] = None

    def add_preprocessing(self, module: PreprocessingModule):
        self.preprocessing_modules.append(module)

    def clear_preprocessing(self):
        self.preprocessing_modules.clear()

    def preprocess_image(self, image: np.ndarray) -> DataObject | List[DataObject]:
        processed = DataObject(image.copy())
        for module in self.preprocessing_modules:
            processed = module.process(processed)
        return processed

    def set_feature_extractor(self, extractor: BaseFeatureExtractor):
        self.feature_extractor = extractor

    def extract_features(self, preprocessed_image: DataObject | List[DataObject]) -> FeatureVector | List[FeatureVector]:
        if self.feature_extractor is None:
            raise RuntimeError("No feature extractor set. Call set_feature_extractor() first.")
        return self.feature_extractor.extract(preprocessed_image)

    def set_classifier(self, classifier: BaseClassifier):
        self.classifier = classifier

    def classify(self, features: FeatureVector | List[FeatureVector]) -> ClassificationResult | List[ClassificationResult]:
        if self.classifier is None:
            raise RuntimeError("No classifier set. Call set_classifier() first.")
        return self.classifier.classify(features)

    def process_full_pipeline(self, image: np.ndarray) -> tuple:
        preprocessed = self.preprocess_image(image)
        features = self.extract_features(preprocessed)
        result = self.classify(features)

        if isinstance(preprocessed, list):
            annotated_single = self.feature_extractor.visualize(preprocessed[0].data, features[0])
            annotated = [annotated_single] * len(preprocessed)
        else:
            annotated = self.feature_extractor.visualize(preprocessed.data, features)

        return preprocessed, features, result, annotated

    def get_pipeline_info(self) -> str:
        parts = []

        if self.preprocessing_modules:
            names = [str(m) for m in self.preprocessing_modules]
            parts.append(f"Preprocessing: {' → '.join(names)}")
        else:
            parts.append("Preprocessing: None")

        if self.feature_extractor:
            parts.append(f"Features: {self.feature_extractor.name}")
        else:
            parts.append("Features: Not set")

        if self.classifier:
            parts.append(f"Classifier: {self.classifier.name}")
        else:
            parts.append("Classifier: Not set")

        return " | ".join(parts)

    def is_pipeline_complete(self) -> bool:
        return (
            self.feature_extractor is not None and
            self.classifier is not None
        )