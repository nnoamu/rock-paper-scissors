"""
Rock-Paper-Scissors - Desktop Application Entry Point
Nativ desktop alkalmazas Dear PyGui-val.

Hasznalat:
    python src/app_desktop.py
"""

import sys
from pathlib import Path

# Src path hozzaadasa
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from core.pipeline import ProcessingPipeline
from preprocessing.grayscale import GrayscaleConverter
from preprocessing.channel_splitter import ChannelSplitter
from feature_extraction.dummy_geometric_extractor import DummyGeometricExtractor
from classification.dummy_classifier import DummyClassifier
from desktop_ui.app import DesktopApp
from game import TwoPlayerGameWrapper


def main():
    """Fo belepesi pont."""
    print("=" * 50)
    print("Rock-Paper-Scissors Desktop App")
    print("=" * 50)

    # Pipeline letrehozasa
    pipeline = ProcessingPipeline()

    # Desktop UI letrehozasa
    app = DesktopApp(pipeline)

    # Preprocessorok regisztralasa
    app.register_preprocessor("None", None)
    app.register_preprocessor("Grayscale", GrayscaleConverter())
    app.register_preprocessor("Channel Split", ChannelSplitter())
    print("[+] Preprocessors registered")

    # Feature extractorok regisztralasa
    app.register_feature_extractor("Geometric (Dummy)", DummyGeometricExtractor())
    print("[+] Feature extractors registered")

    # Classifierek regisztralasa
    app.register_classifier("Simple Dummy", DummyClassifier())
    print("[+] Classifiers registered")

    # Game wrapper (ketjatekos mod)
    game_wrapper = TwoPlayerGameWrapper(pipeline, min_confidence=0.7)
    app.set_game_wrapper(game_wrapper)
    print("[+] Two-player game mode enabled")

    print("\n" + "=" * 50)
    print("Starting desktop application...")
    print("=" * 50 + "\n")

    # Alkalmazas inditasa
    app.run()


if __name__ == '__main__':
    main()
