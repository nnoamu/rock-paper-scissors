"""
Rock-Paper-Scissors - Main Application Entry Point
"""

import sys
from pathlib import Path

src_path = Path(__file__)
sys.path.insert(0, str(src_path))

from core.pipe_network import ProcessingPipeNetwork
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.channel_splitter import ChannelSplitter
from preprocessing import (
    DownscaleModule,
    SkinColorSegmenterModule, 
    GaussianBlurModule, 
    EdgeDetectorModule, 
    DownscaleWithInterpolationPreprocessor, 
    ThresholdFillModule, 
    HoleClosingModule,
    EdgeSmoothingModule,
    ObjectSeparatorModule
    )
from feature_extraction.dummy_geometric_extractor import DummyGeometricExtractor
from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor
from feature_extraction.mediapipe_enhanced_extractor import MediaPipeEnhancedExtractor
from feature_extraction.hu_moments_extractor import HuMomentsExtractor
from classification.dummy_classifier import DummyClassifier
from classification.mediapipe_rf_classifier import MediaPipeRFClassifier
from classification.mediapipe_enhanced_rf_classifier import MediaPipeEnhancedRFClassifier
from classification.rule_based_classifier import RuleBasedGestureClassifier
from classification.kmeans_classifier import KMeansClassifier
from ui.main_interface import MainInterface
from game import TwoPlayerGameWrapper
from ui.styles.custom_css import get_custom_css
import gradio as gr

def main():
    pipeline = ProcessingPipeNetwork()
    interface = MainInterface(pipeline)

    interface.register_preprocessor("None", None)
    interface.register_preprocessor("Downscale (640px)", DownscaleWithInterpolationPreprocessor(max_size=640))
    interface.register_preprocessor("Downscale (480px)", DownscaleWithInterpolationPreprocessor(max_size=480))
    interface.register_preprocessor("Downscale (300px)", DownscaleModule(max_dimension_length=300))
    interface.register_preprocessor("Grayscale", GrayscaleConverter())
    interface.register_preprocessor("Split", ChannelSplitter())

    # Skin color segmentation models (optional - require trained models)
    try:
        interface.register_preprocessor("Skin color based (<400)", SkinColorSegmenterModule('models/skin_segmentation/model1'))
    except Exception:
        print("Warning: Skin color segmentation model1 not found, skipping...")

    try:
        interface.register_preprocessor("Skin color based (<14000)", SkinColorSegmenterModule('models/skin_segmentation/model2'))
    except Exception:
        print("Warning: Skin color segmentation model2 not found, skipping...")

    interface.register_preprocessor("Blur", GaussianBlurModule())
    interface.register_preprocessor("Edge detection", EdgeDetectorModule(lower_thresh=0, upper_thresh=40))
    interface.register_preprocessor("Threshold fill", ThresholdFillModule(), deps=['GaussianBlur', 'EdgeDetector'])
    interface.register_preprocessor("Hole closing", HoleClosingModule())
    interface.register_preprocessor("Edge smoothing", EdgeSmoothingModule())
    interface.register_preprocessor("Object separation", ObjectSeparatorModule())


    # Feature Extractors
    interface.register_feature_extractor("Geometric (Dummy)", DummyGeometricExtractor())
    interface.register_feature_extractor("MediaPipe Hand Landmarks", MediaPipeHandExtractor())
    interface.register_feature_extractor("MediaPipe Enhanced (91D)", MediaPipeEnhancedExtractor())
    interface.register_feature_extractor("Hu moments extractor", HuMomentsExtractor())

    # Classifiers
    interface.register_classifier("Simple Dummy", DummyClassifier())
    interface.register_classifier("Rule-Based (No Training)", RuleBasedGestureClassifier())
    interface.register_classifier("K nearest classifier", KMeansClassifier(k=5, data_path=[
        'data/processed/drgfreeman/hu_moments.pkl', 
        'data/processed/hectorandac/hu_moments.pkl'
        ]))

    # 63D classifier
    try:
        rf_classifier = MediaPipeRFClassifier()
        rf_classifier.load_model()
        interface.register_classifier("Random Forest 63D", rf_classifier)
    except FileNotFoundError:
        print("Warning: MediaPipe RF 63D model not found.")

    # 91D classifier
    try:
        rf_enhanced_classifier = MediaPipeEnhancedRFClassifier()
        rf_enhanced_classifier.load_model()
        interface.register_classifier("Random Forest 91D", rf_enhanced_classifier)
    except FileNotFoundError:
        print("Warning: MediaPipe RF 91D model not found.")

    game_wrapper = TwoPlayerGameWrapper(pipeline, min_confidence=0.7)
    interface.set_game_wrapper(game_wrapper)
    print("✅ Two-player game mode enabled")

    demo = interface.create_interface()

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=get_custom_css()
    )


if __name__ == '__main__':
    main()