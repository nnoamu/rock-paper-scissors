"""
Fo Dear PyGui desktop alkalmazas.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from typing import Dict, Optional
import time

from desktop_ui.theme import (
    create_theme,
    create_button_green_theme,
    Colors,
    rgb_to_dpg
)
from desktop_ui.components.camera import CameraManager
from desktop_ui.components.header import HeaderPanel
from desktop_ui.components.pipeline_panel import PipelinePanel
from desktop_ui.components.result_panel import ResultPanel
from desktop_ui.components.preview import PreviewPanel
from desktop_ui.components.debug_log import DebugLogPanel, PipelineInfoPanel
from desktop_ui.utils.image_utils import create_placeholder_image


class DesktopApp:
    """
    Fo desktop alkalmazas osztaly.
    Osszekapcsolja a pipeline-t a Dear PyGui UI-val.
    """

    # Ablak meretek
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.game_wrapper = None

        # Regisztralt modulok
        self.preprocessors: Dict[str, Optional[object]] = {}
        self.feature_extractors: Dict[str, Optional[object]] = {}
        self.classifiers: Dict[str, Optional[object]] = {}

        # Aktualis kepek
        self.current_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None

        # Komponensek
        self.camera = CameraManager()
        self.header = HeaderPanel()
        self.pipeline_panel = PipelinePanel()
        self.result_panel = ResultPanel()
        self.preview_panel = PreviewPanel(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        self.debug_log = DebugLogPanel()
        self.pipeline_info = PipelineInfoPanel()

        # Kamera allapot
        self.camera_active = False
        self.live_mode = False

        # Process gomb tema
        self.green_button_theme = None

    def set_game_wrapper(self, game_wrapper):
        """Game wrapper beallitasa."""
        self.game_wrapper = game_wrapper

    def register_preprocessor(self, name: str, module):
        """Preprocessor regisztralasa."""
        self.preprocessors[name] = module

    def register_feature_extractor(self, name: str, extractor):
        """Feature extractor regisztralasa."""
        self.feature_extractors[name] = extractor

    def register_classifier(self, name: str, classifier):
        """Classifier regisztralasa."""
        self.classifiers[name] = classifier

    def _setup_dpg(self):
        """Dear PyGui inicializalasa."""
        dpg.create_context()

        # Tema alkalmazasa
        global_theme = create_theme()
        dpg.bind_theme(global_theme)

        self.green_button_theme = create_button_green_theme()

        # Viewport (fo ablak)
        dpg.create_viewport(
            title="Rock-Paper-Scissors - Desktop",
            width=self.WINDOW_WIDTH,
            height=self.WINDOW_HEIGHT,
            min_width=1200,
            min_height=700
        )

    def _create_ui(self):
        """UI feluletek letrehozasa."""
        with dpg.window(tag="main_window"):

            # === HEADER ===
            self.header.create("main_window", self.WINDOW_WIDTH - 30)
            dpg.add_spacer(height=15)

            # === FO TARTALOM ===
            with dpg.group(horizontal=True):

                # === BAL OLDAL: Kamera + Kontrollok ===
                with dpg.child_window(width=380, height=700, border=False):

                    # Kamera kontrollok
                    with dpg.child_window(width=360, height=120, border=True, tag="camera_controls"):
                        from desktop_ui.theme import create_panel_theme
                        dpg.bind_item_theme("camera_controls", create_panel_theme())

                        dpg.add_spacer(height=5)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=10)
                            dpg.add_text("  CAMERA", color=rgb_to_dpg(Colors.PRIMARY))

                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=10)
                            dpg.add_button(
                                label="Start Camera",
                                width=150,
                                callback=self._toggle_camera,
                                tag="camera_btn"
                            )
                            dpg.add_spacer(width=10)
                            dpg.add_button(
                                label="Capture",
                                width=100,
                                callback=self._capture_frame,
                                tag="capture_btn"
                            )

                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=10)
                            dpg.add_checkbox(
                                label="Live Processing",
                                default_value=False,
                                callback=self._toggle_live_mode,
                                tag="live_checkbox"
                            )
                            dpg.add_spacer(width=20)
                            dpg.add_text("FPS: -", tag="fps_text",
                                        color=rgb_to_dpg(Colors.TEXT_SECONDARY))

                    dpg.add_spacer(height=15)

                    # Kep betoltese fajlbol
                    with dpg.child_window(width=360, height=80, border=True, tag="file_controls"):
                        from desktop_ui.theme import create_panel_theme
                        dpg.bind_item_theme("file_controls", create_panel_theme())

                        dpg.add_spacer(height=5)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=10)
                            dpg.add_text("  LOAD IMAGE", color=rgb_to_dpg(Colors.PRIMARY))

                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=10)
                            dpg.add_button(
                                label="Browse...",
                                width=150,
                                callback=self._show_file_dialog
                            )

                    dpg.add_spacer(height=15)

                    # Process gomb
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=10)
                        process_btn = dpg.add_button(
                            label="  PROCESS IMAGE",
                            width=340,
                            height=50,
                            callback=self._process_image
                        )
                        dpg.bind_item_theme(process_btn, self.green_button_theme)

                    dpg.add_spacer(height=15)

                    # Eredmeny panel
                    self.result_panel.create("main_window", 360)

                    dpg.add_spacer(height=15)

                    # Pipeline info
                    self.pipeline_info.create("main_window", 360, 80)

                dpg.add_spacer(width=20)

                # === KOZEP: Pipeline + Preview ===
                with dpg.child_window(width=750, height=700, border=False):

                    # Pipeline valasztok
                    self.pipeline_panel.set_options(
                        list(self.preprocessors.keys()),
                        list(self.feature_extractors.keys()),
                        list(self.classifiers.keys())
                    )
                    self.pipeline_panel.create("main_window", 730)

                    dpg.add_spacer(height=15)

                    # Preview panel
                    self.preview_panel.create("main_window", 730)

                dpg.add_spacer(width=20)

                # === JOBB OLDAL: Debug Log ===
                with dpg.child_window(width=280, height=700, border=False):
                    self.debug_log.create("main_window", 260, 680)

        # File dialog (rejtett)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_selected,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".bmp", color=(0, 255, 0, 255))

        dpg.set_primary_window("main_window", True)

    def _toggle_camera(self, sender, app_data):
        """Kamera be/ki kapcsolasa."""
        if self.camera_active:
            self.camera.stop()
            self.camera_active = False
            dpg.set_item_label("camera_btn", "Start Camera")
            dpg.set_value("fps_text", "FPS: -")
        else:
            if self.camera.start():
                self.camera_active = True
                dpg.set_item_label("camera_btn", "Stop Camera")

    def _toggle_live_mode(self, sender, app_data):
        """Elo feldolgozas be/ki."""
        self.live_mode = app_data

    def _capture_frame(self, sender, app_data):
        """Pillanatkep keszitese kamerabol."""
        if self.camera_active:
            frame = self.camera.capture_snapshot()
            if frame is not None:
                self.current_image = frame
                self.preview_panel.set_images(frame, None, None)
                self.debug_log.append("[Camera] Frame captured")

    def _show_file_dialog(self, sender, app_data):
        """Fajl megnyitas dialog."""
        dpg.show_item("file_dialog")

    def _on_file_selected(self, sender, app_data):
        """Fajl kivalasztasakor."""
        if app_data and 'file_path_name' in app_data:
            file_path = app_data['file_path_name']
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.preview_panel.set_images(image, None, None)
                self.debug_log.append(f"[File] Loaded: {file_path}")
            else:
                self.debug_log.append(f"[Error] Could not load: {file_path}")

    def _process_image(self, sender=None, app_data=None):
        """Kep feldolgozasa a pipeline-nal."""
        if self.current_image is None:
            self.debug_log.append("[Info] No image to process")
            return

        game_mode = self.pipeline_panel.get_game_mode()

        if game_mode == "Two Player Game":
            self._process_game_mode()
        else:
            self._process_single_mode()

    def _process_single_mode(self):
        """Egyjatekos mod feldolgozasa."""
        self.debug_log.clear()

        preproc_name = self.pipeline_panel.get_preprocessing()
        feature_name = self.pipeline_panel.get_feature_extractor()
        classifier_name = self.pipeline_panel.get_classifier()

        # Pipeline konfiguralas
        self.pipeline.clear_preprocessing()
        if preproc_name in self.preprocessors and self.preprocessors[preproc_name] is not None:
            self.pipeline.add_preprocessing(self.preprocessors[preproc_name])
            self.debug_log.append(f"[Stage 1] Preprocessing: {preproc_name}")
        else:
            self.debug_log.append("[Stage 1] Preprocessing: None")

        if feature_name not in self.feature_extractors:
            self.debug_log.append("[Stage 2] ERROR: No feature extractor")
            self.result_panel.set_error()
            return

        self.pipeline.set_feature_extractor(self.feature_extractors[feature_name])
        self.debug_log.append(f"[Stage 2] Feature Extractor: {feature_name}")

        if classifier_name not in self.classifiers:
            self.debug_log.append("[Stage 3] ERROR: No classifier")
            self.result_panel.set_error()
            return

        self.pipeline.set_classifier(self.classifiers[classifier_name])
        self.debug_log.append(f"[Stage 3] Classifier: {classifier_name}")

        try:
            self.debug_log.append("\n[Pipeline] Starting...")

            preprocessed, features, result, annotated = self.pipeline.process_full_pipeline(
                self.current_image
            )

            # Lista kezeles (batch output)
            if isinstance(preprocessed, list):
                preprocessed = preprocessed[0].data
            else:
                preprocessed = preprocessed.data
            if isinstance(features, list):
                features = features[0]
            if isinstance(result, list):
                result = result[0]

            self.processed_image = preprocessed
            self.annotated_image = annotated

            # Feature info
            self.debug_log.append(f"\n[Features] Type: {features.feature_type.value}")
            self.debug_log.append(f"[Features] Dim: {features.feature_dimension}")
            self.debug_log.append(f"[Features] Time: {features.extraction_time_ms:.2f}ms")

            # Eredmeny
            self.debug_log.append(f"\n[Result] Class: {result.predicted_class.value.upper()}")
            self.debug_log.append(f"[Result] Confidence: {result.get_confidence_percentage():.1f}%")

            self.result_panel.update(
                result.predicted_class.value,
                result.get_confidence_percentage()
            )

            # Preview frissites
            self.preview_panel.set_images(
                self.current_image,
                self.processed_image,
                self.annotated_image
            )

            # Pipeline info
            self.pipeline_info.update(self.pipeline.get_pipeline_info())

            self.debug_log.append("\n[Pipeline] Complete!")

        except Exception as e:
            self.debug_log.append(f"\n[ERROR] {str(e)}")
            self.result_panel.set_error()

    def _process_game_mode(self):
        """Ketjatekos mod feldolgozasa."""
        self.debug_log.clear()

        if self.game_wrapper is None:
            self.debug_log.append("[ERROR] Game wrapper not initialized")
            self.result_panel.set_error()
            return

        preproc_name = self.pipeline_panel.get_preprocessing()
        feature_name = self.pipeline_panel.get_feature_extractor()
        classifier_name = self.pipeline_panel.get_classifier()

        # Pipeline konfiguralas
        self.pipeline.clear_preprocessing()

        from preprocessing.image_splitter import ImageSplitterModule
        self.pipeline.add_preprocessing(ImageSplitterModule())
        self.debug_log.append("[Stage 1a] Image Splitter: 2-player batch")

        if preproc_name in self.preprocessors and self.preprocessors[preproc_name] is not None:
            self.pipeline.add_preprocessing(self.preprocessors[preproc_name])
            self.debug_log.append(f"[Stage 1b] Additional: {preproc_name}")

        if feature_name not in self.feature_extractors:
            self.debug_log.append("[Stage 2] ERROR: No feature extractor")
            self.result_panel.set_error()
            return

        self.pipeline.set_feature_extractor(self.feature_extractors[feature_name])
        self.debug_log.append(f"[Stage 2] Feature Extractor: {feature_name}")

        if classifier_name not in self.classifiers:
            self.debug_log.append("[Stage 3] ERROR: No classifier")
            self.result_panel.set_error()
            return

        self.pipeline.set_classifier(self.classifiers[classifier_name])
        self.debug_log.append(f"[Stage 3] Classifier: {classifier_name}")

        try:
            self.debug_log.append("\n[Game Mode] Running evaluation...")

            game_result, annotated = self.game_wrapper.run_with_visualization(self.current_image)

            self.annotated_image = annotated

            p1 = game_result.player1_result
            p2 = game_result.player2_result

            self.debug_log.append(f"\n[Player 1] {p1.predicted_class.value.upper()} ({p1.get_confidence_percentage():.1f}%)")
            self.debug_log.append(f"[Player 2] {p2.predicted_class.value.upper()} ({p2.get_confidence_percentage():.1f}%)")
            self.debug_log.append(f"\n[Game] Status: {game_result.status.upper()}")

            # Eredmeny formatalas
            p1_class = p1.predicted_class.value.upper()
            p2_class = p2.predicted_class.value.upper()

            if game_result.status == 'invalid':
                result_text = f"Invalid: {p1_class} vs {p2_class}"
                details = game_result.reason or ""
            elif game_result.status == 'draw':
                result_text = f"Draw! {p1_class} vs {p2_class}"
                details = f"P1: {p1.get_confidence_percentage():.0f}% | P2: {p2.get_confidence_percentage():.0f}%"
            else:
                winner = "P1" if game_result.winner == 'player1' else "P2"
                result_text = f"{winner} Wins! {p1_class} vs {p2_class}"
                details = f"P1: {p1.get_confidence_percentage():.0f}% | P2: {p2.get_confidence_percentage():.0f}%"

            self.result_panel.update_game_result(result_text, details)

            # Preview
            self.preview_panel.set_images(
                self.current_image,
                None,
                self.annotated_image
            )

            self.pipeline_info.update(self.pipeline.get_pipeline_info())
            self.debug_log.append("\n[Game Mode] Complete!")

        except Exception as e:
            self.debug_log.append(f"\n[ERROR] {str(e)}")
            import traceback
            self.debug_log.append(traceback.format_exc())
            self.result_panel.set_error()

    def _update_loop(self):
        """Fo update ciklus (kamera frissites, stb)."""
        if self.camera_active:
            frame = self.camera.get_frame()
            if frame is not None:
                # FPS kijelzes
                fps = self.camera.get_fps()
                dpg.set_value("fps_text", f"FPS: {fps:.1f}")

                # Live mode: folyamatos feldolgozas
                if self.live_mode:
                    self.current_image = frame
                    self._process_image()
                else:
                    # Csak preview frissites
                    self.preview_panel.update_live(frame)

    def run(self):
        """Alkalmazas futtatasa."""
        self._setup_dpg()
        self._create_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Fo ciklus
        while dpg.is_dearpygui_running():
            self._update_loop()
            dpg.render_dearpygui_frame()

        # Tisztitas
        self.camera.stop()
        dpg.destroy_context()
