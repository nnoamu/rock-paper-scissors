"""
Háló struktúrájú feldolgozó pipeline.

Fázisok:
1. Preprocessing - Képjavítás (több modul)
2. Feature Extraction - Jellemzők kinyerése (1 modul)
3. Classification - Döntéshozatal (1 modul)

Preprocessing és Feature Extraction moduloknál beállítható név és függőségi lista:
- ha nincs megadva név, akkor a modul name attribútuma lesz a név
- ha nincs megadva függőség, akkor a preproc listához utoljára hozzáadott modul outputját köti be, vagy az input képet, ha üres a lista

Példa használat:
    pipe=ProcessingPipeNetwork()

    pipe.add_preprocessing(SkinColorSegmenterModule('models/skin_segmentation/model1'), 'segmenter')
    pipe.add_preprocessing(GaussianBlurModule(), 'blur', deps='segmenter')
    pipe.add_preprocessing(EdgeDetectorModule(lower_thresh=0, upper_thresh=40))
    pipe.set_feature_extractor(DummyGeometricExtractor(), deps=['blur'])
    pipe.set_classifier(DummyClassifier())

    data=cv2.imread('src/img.png')
    data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    pipe.process_full_pipeline(data)

    cv2.imshow("seg", pipe.get_preprocessed_image('segmenter').data)
    cv2.imshow("blur", pipe.get_preprocessed_image('blurr').data)
    cv2.imshow("edge", pipe.get_preprocessed_image('EdgeDetector').data)
"""

from typing import List, Dict, Optional
import numpy as np

from .base_module import BaseModule
from .base_processor import PreprocessingModule
from .base_feature_extractor import BaseFeatureExtractor
from .base_classifier import BaseClassifier
from .feature_vector import FeatureVector
from .classification_result import ClassificationResult
from .data_object import DataObject

class ProcessingPipeNetwork:

    def __init__(self):
        self.preprocessing_modules: Dict[str, BaseModule]={}
        self.preprocessing_outputs: Dict[str, DataObject | List[DataObject]]={}
        self.preprocessing_order: List[str]=[]

        self.feature_extractor: Optional[BaseFeatureExtractor]=None
        self.feature_extractor_name: Optional[str]=None
        self.feature_output: Optional[FeatureVector | List[FeatureVector]]=None

        self.classifier: Optional[BaseClassifier]=None
        self.classifier_output: Optional[ClassificationResult | List[ClassificationResult]]=None

        self.dependencies: Dict[str, str | List[str]]={}

    def add_preprocessing(self, module: PreprocessingModule, name: Optional[str]=None, deps: Optional[str | List[str]]=None):
        if name is None:
            name=module.name
        
        if name in self.preprocessing_modules:
            raise KeyError("Pipe network error: module with name '"+name+"' already exists.")
        
        if deps is None:
            if len(self.preprocessing_order)>0:
                deps=self.preprocessing_order[-1]
            else:
                deps='input'
        elif isinstance(deps, list):
            for dep in deps:
                if (dep not in self.preprocessing_modules) and dep!='input':
                    raise KeyError("Pipe network error: module dependency '"+dep+"' doesn't exist.")
        else: #str
            if (deps not in self.preprocessing_modules) and deps!='input':
                raise KeyError("Pipe network error: module dependency '"+deps+"' doesn't exist.")
        
        self.preprocessing_modules[name]=module
        self.dependencies[name]=deps
        self.preprocessing_order.append(name)

    def clear_preprocessing(self):
        for name in self.preprocessing_order:
            self.preprocessing_modules.pop(name)
            self.dependencies.pop(name)
            self.preprocessing_outputs.pop(name)
        self.preprocessing_order.clear()
    
    def __combine_input_objects(self, inputs: List[str]) -> DataObject | List[DataObject]:
        if len(inputs)==1:
            return self.preprocessing_outputs[inputs[0]]

        if isinstance(self.preprocessing_outputs[inputs[0]], DataObject):
            shape=self.preprocessing_outputs[inputs[0]].data.shape
            data=[]
            for name in inputs:
                current=self.preprocessing_outputs[name].data

                if isinstance(current, list):
                    raise RuntimeError("Pipe network error: Received singular and list inputs at the same time.")
                if current.shape!=shape:
                    raise RuntimeError("Pipe network error: The received inputs don't have the same shape.")
                data.append(current)
            
            data=np.array(data)
            return DataObject(data)

        #else: List[DataObject]
        l=len(self.preprocessing_outputs[inputs[0]])
        for name in inputs:
            if isinstance(current, DataObject):
                raise RuntimeError("Pipe network error: Received singular and list inputs at the same time.")
            if l!=len(self.preprocessing_outputs[name]):
                raise RuntimeError("Pipe network error: The received input lists don't have the same length.")
        
        result=[]
        for i in range(l):
            shape=self.preprocessing_outputs[inputs[0]][i].data.shape
            data=[]

            for name in inputs:
                current=self.preprocessing_outputs[name][i].data
                if current.shape!=shape:
                    raise RuntimeError("Pipe network error: The received inputs don't have the same shape.")
                data.append(current)

            data=np.array(data)
            result.append(DataObject(data))
        
        return result

    def preprocess_image(self, image: np.ndarray):
        processed = DataObject(image.copy())
        self.preprocessing_outputs['input']=processed
        for name in self.preprocessing_order:
            module=self.preprocessing_modules[name]
            input=self.dependencies[name]
            if isinstance(input, list):
                input=self.__combine_input_objects(input)
            else:
                input=self.preprocessing_outputs[input]
        
            self.preprocessing_outputs[name]=module.process(input)

    def set_feature_extractor(self, extractor: BaseFeatureExtractor, name: Optional[str]=None, deps: Optional[str | List[str]]=None):
        if name is None:
            name=extractor.name

        if self.feature_extractor_name is not None:
            self.dependencies.pop(name)

        if name in self.dependencies:
            raise KeyError("module with name '"+name+"' already exists.")

        if deps is None:
            if len(self.preprocessing_order)>0:
                deps=self.preprocessing_order[-1]
            else:
                deps='input'
        elif isinstance(deps, list):
            for dep in deps:
                if (dep not in self.preprocessing_modules) and dep!='input':
                    raise KeyError("Pipe network error: module dependency '"+dep+"' doesn't exist.")
        else: #str
            if (deps not in self.preprocessing_modules) and deps!='input':
                raise KeyError("Pipe network error: module dependency '"+deps+"' doesn't exist.")

        self.feature_extractor_name=name
        self.feature_extractor=extractor
        self.dependencies[name]=deps

    def extract_features(self):
        if (self.feature_extractor is None) or (self.feature_extractor_name is None) or (self.dependencies[self.feature_extractor_name] is None):
            raise RuntimeError("No feature extractor set. Call set_feature_extractor() first.")
        
        input=self.dependencies[self.feature_extractor_name]
        if isinstance(input, list):
            input=self.__combine_input_objects(input)
        else:
            input=self.preprocessing_outputs[input]
    
        self.feature_output=self.feature_extractor.extract(input)

    def set_classifier(self, classifier: BaseClassifier):
        self.classifier=classifier

    def classify(self):
        if self.classifier is None:
            raise RuntimeError("No classifier set. Call set_classifier() first.")
        if self.feature_output is None:
            raise RuntimeError("No feature vector available. Call extract_features() first.")
        
        self.classifier_output=self.classifier.classify(self.feature_output)

    def process_full_pipeline(self, image: np.ndarray) -> tuple:
        if self.feature_extractor is None:
            raise RuntimeError("No feature extractor set. Call set_feature_extractor() first.")
        if self.classifier is None:
            raise RuntimeError("No classifier set. Call set_classifier() first.")

        self.preprocessing_outputs.clear()

        self.preprocess_image(image)
        self.extract_features()
        self.classify()

        # ================ Temp megoldas ================
        # kompatibilitás miatt készül itt is visszatérési érték
        # ha "bármi" outputját el akarjuk érni, arra ott vannak a getterek (ld. lentebb)

        # csak az utolsó preproc. modul outputját adja vissza
        preprocessed=self.preprocessing_outputs[self.preprocessing_order[-1]]
        features=self.feature_output
        result=self.classifier_output

        if isinstance(preprocessed, list):
            if (not isinstance(features, list)) or len(preprocessed)!=len(features):
                raise RuntimeError("Pipe network error: Annotation: preprocessing and feature list lengths differ.")
            annotated = [
                self.feature_extractor.visualize(preprocessed[i].data, features[i])
                for i in range(len(preprocessed))
            ]
        else:
            if isinstance(features, list) or (features is None) or (preprocessed is None):
                raise RuntimeError("Pipe network error: Annotation: preprocessing and feature list lengths differ.")
            annotated = self.feature_extractor.visualize(preprocessed.data, features)

        return preprocessed, features, result, annotated
        # ============= Temp megoldas vege ==============

    def get_preprocessed_image(self, name: str) -> DataObject | List[DataObject]:
        if name not in self.preprocessing_modules:
            raise KeyError("Pipe network error: no module named '"+name+"' exists.")
        return self.preprocessing_outputs[name]
    
    def get_feature_vector(self) -> Optional[FeatureVector | List[FeatureVector]]:
        return self.feature_output
    
    def get_annotated_image(self, name: str) -> np.ndarray | List[np.ndarray]:
        if self.feature_extractor is None:
            raise RuntimeError("No feature extractor set. Call set_feature_extractor() first.")
        if name not in self.preprocessing_modules:
            raise RuntimeError("Pipe network error: no module named '"+name+"' exists.")
        
        image=self.preprocessing_outputs[name]
        features=self.feature_output
        if isinstance(image, list):
            if (not isinstance(features, list)) or len(features)!=len(image):
                raise RuntimeError("Pipe network error: Annotation: preprocessing and feature list lengths differ.")
            annotated = [
                self.feature_extractor.visualize(image[i].data, features[i])
                for i in range(len(image))
            ]
        else:
            if isinstance(features, list) or (features is None):
                raise RuntimeError("Pipe network error: Annotation: preprocessing and feature list lengths differ.")
            annotated = self.feature_extractor.visualize(image.data, features)
        
        return annotated
    
    def get_result(self) -> Optional[ClassificationResult | List[ClassificationResult]]:
        return self.classifier_output

    def get_pipeline_info(self) -> str:
        parts = []

        if self.preprocessing_modules:
            names = [module.name for _, module in self.preprocessing_modules.items()]
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