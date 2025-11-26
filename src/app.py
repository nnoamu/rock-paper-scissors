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
from preprocessing import SkinColorSegmenterModule, GaussianBlurModule, EdgeDetectorModule
from feature_extraction.dummy_geometric_extractor import DummyGeometricExtractor
from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor
from classification.dummy_classifier import DummyClassifier
from classification.random_forest_classifier import MediaPipeRFClassifier
from ui.main_interface import MainInterface
from game import TwoPlayerGameWrapper


def main():
    pipeline = ProcessingPipeNetwork()
    interface = MainInterface(pipeline)

    # Preprocessors
    interface.register_preprocessor("None", None)
    interface.register_preprocessor("Grayscale", GrayscaleConverter())
    interface.register_preprocessor("Split", ChannelSplitter())
    interface.register_preprocessor("Skin color based (<400)", SkinColorSegmenterModule('models/skin_segmentation/model1'))
    interface.register_preprocessor("Skin color based (<14000)", SkinColorSegmenterModule('models/skin_segmentation/model2'))
    interface.register_preprocessor("Blur", GaussianBlurModule())
    interface.register_preprocessor("Edge detection", EdgeDetectorModule(lower_thresh=0, upper_thresh=40))

    # Feature Extractors
    interface.register_feature_extractor("Geometric (Dummy)", DummyGeometricExtractor())
    interface.register_feature_extractor("MediaPipe Hand Landmarks", MediaPipeHandExtractor())

    # Classifiers
    interface.register_classifier("Simple Dummy", DummyClassifier())

    # MediaPipe RF Classifier (load model if exists)
    try:
        rf_classifier = MediaPipeRFClassifier()
        rf_classifier.load_model()
        interface.register_classifier("Random Forest (MediaPipe)", rf_classifier)
    except FileNotFoundError:
        print("⚠️  MediaPipe RF model not found. Train it with: python scripts/train_mediapipe_rf.py")

    game_wrapper = TwoPlayerGameWrapper(pipeline, min_confidence=0.7)
    interface.set_game_wrapper(game_wrapper)
    print("✅ Two-player game mode enabled")

    demo = interface.create_interface()

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == '__main__':
    main()