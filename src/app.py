"""
Rock-Paper-Scissors - Main Application Entry Point
"""

import sys
from pathlib import Path

print("🚀 Starting application...")

src_path = Path(__file__)
sys.path.insert(0, str(src_path))

print("📦 Importing modules...")

from core.pipe_network import ProcessingPipeNetwork
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.channel_splitter import ChannelSplitter
from preprocessing import SkinColorSegmenterModule, GaussianBlurModule, EdgeDetectorModule, DownscaleWithInterpolationPreprocessor
from feature_extraction.dummy_geometric_extractor import DummyGeometricExtractor
from feature_extraction.mediapipe_hand_extractor import MediaPipeHandExtractor
from feature_extraction.mediapipe_enhanced_extractor import MediaPipeEnhancedExtractor
from classification.dummy_classifier import DummyClassifier
from classification.mediapipe_rf_classifier import MediaPipeRFClassifier
from classification.mediapipe_enhanced_rf_classifier import MediaPipeEnhancedRFClassifier
from classification.rule_based_classifier import RuleBasedGestureClassifier
from ui.main_interface import MainInterface
from game import TwoPlayerGameWrapper
from ui.styles.custom_css import get_custom_css
import gradio as gr

print("✅ All imports successful")

def main():
    print("🔧 Initializing pipeline...")
    pipeline = ProcessingPipeNetwork()
    interface = MainInterface(pipeline)

    print("📝 Registering preprocessors...")
    interface.register_preprocessor("None", None)
    interface.register_preprocessor("Downscale (640px)", DownscaleWithInterpolationPreprocessor(max_size=640))
    interface.register_preprocessor("Downscale (480px)", DownscaleWithInterpolationPreprocessor(max_size=480))
    interface.register_preprocessor("Grayscale", GrayscaleConverter())
    interface.register_preprocessor("Split", ChannelSplitter())

    # Skin color segmentation models (optional - require trained models)
    try:
        interface.register_preprocessor("Skin color based (<400)", SkinColorSegmenterModule('models/skin_segmentation/model1'))
    except Exception as e:
        print(f"⚠️ Skin color segmentation model1 not found: {e}")

    try:
        interface.register_preprocessor("Skin color based (<14000)", SkinColorSegmenterModule('models/skin_segmentation/model2'))
    except Exception as e:
        print(f"⚠️ Skin color segmentation model2 not found: {e}")

    interface.register_preprocessor("Blur", GaussianBlurModule())
    interface.register_preprocessor("Edge detection", EdgeDetectorModule(lower_thresh=0, upper_thresh=40))

    print("🎯 Registering feature extractors...")
    # Feature Extractors
    interface.register_feature_extractor("Geometric (Dummy)", DummyGeometricExtractor())
    interface.register_feature_extractor("MediaPipe Hand Landmarks", MediaPipeHandExtractor())
    interface.register_feature_extractor("MediaPipe Enhanced (91D)", MediaPipeEnhancedExtractor())

    print("🤖 Registering classifiers...")
    # Classifiers
    interface.register_classifier("Simple Dummy", DummyClassifier())
    interface.register_classifier("Rule-Based (No Training)", RuleBasedGestureClassifier())

    # 63D classifier
    try:
        rf_classifier = MediaPipeRFClassifier()
        rf_classifier.load_model()
        interface.register_classifier("Random Forest 63D", rf_classifier)
    except FileNotFoundError as e:
        print(f"⚠️ MediaPipe RF 63D model not found: {e}")

    # 91D classifier
    try:
        rf_enhanced_classifier = MediaPipeEnhancedRFClassifier()
        rf_enhanced_classifier.load_model()
        interface.register_classifier("Random Forest 91D", rf_enhanced_classifier)
    except FileNotFoundError as e:
        print(f"⚠️ MediaPipe RF 91D model not found: {e}")

    print("🎮 Setting up game wrapper...")
    game_wrapper = TwoPlayerGameWrapper(pipeline, min_confidence=0.7)
    interface.set_game_wrapper(game_wrapper)
    print("✅ Two-player game mode enabled")

    print("🎨 Creating interface...")
    demo = interface.create_interface()

    print("🚀 Launching Gradio on 0.0.0.0:7860...")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=get_custom_css()
    )

    print("✅ Gradio launched successfully!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)