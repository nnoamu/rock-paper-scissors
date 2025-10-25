"""
Rock-Paper-Scissors - Main Application Entry Point
"""

import sys
from pathlib import Path

src_path = Path(__file__)
sys.path.insert(0, str(src_path))

from core.pipeline import ProcessingPipeline
from preprocessing.grayscale import GrayscaleConverter
from feature_extraction.dummy_geometric_extractor import DummyGeometricExtractor
from classification.dummy_classifier import DummyClassifier
from ui.main_interface import MainInterface
from game import TwoPlayerGameWrapper


def main():
    pipeline = ProcessingPipeline()
    interface = MainInterface(pipeline)

    interface.register_preprocessor("None", None)
    interface.register_preprocessor("Grayscale", GrayscaleConverter())

    interface.register_feature_extractor("Geometric (Dummy)", DummyGeometricExtractor())

    interface.register_classifier("Simple Dummy", DummyClassifier())

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