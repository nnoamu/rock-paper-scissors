"""
Osztályozó modulok ősosztálya.
Tartalmazott korlátozások:
- ClassConstraint(FeatureVector)

Példa használat:
    class MyClassifier(BaseClassifier):
        def _process(self, input: DataObject) -> ClassificationResult:
            return ClassificationResult(
                predicted_class=GestureClass.ROCK,
                confidence=0.95,
                class_probabilities={...},
                processing_time_ms=5.2,
                classifier_name=self.name
            )
"""

from .data_object import DataObject
from .feature_vector import FeatureVector
from .classification_result import ClassificationResult
from .base_module import BaseModule
from constraints import ClassConstraint
from typing import List, cast, final


class BaseClassifier(BaseModule):

    def __init__(self, name: str):
        super().__init__(name)
        self.add_constraint(ClassConstraint(FeatureVector))

        self.is_trained = True

    @final
    def classify(self, features: FeatureVector | List[FeatureVector]) -> ClassificationResult | List[ClassificationResult]:
        # this is fine; az input tényleg DataObject, mert constraint miatt elvárt, hogy FeatureVector, ami a DataObject-ből származik
        result=self.process(cast(DataObject | List[DataObject], features))

        if (not isinstance(result, ClassificationResult)) and ((not isinstance(result, list)) or not isinstance(result[0], ClassificationResult)):
            raise Exception("Error in "+self.name+": Output type is not ClassificationResult or List[ClassificationResult]")
        return cast(ClassificationResult | List[ClassificationResult], result)

    def __str__(self) -> str:
        return self.name