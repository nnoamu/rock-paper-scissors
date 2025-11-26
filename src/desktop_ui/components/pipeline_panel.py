"""
Pipeline valaszto panel komponens.
3 lepcso: Preprocessing -> Feature Extraction -> Classification
"""

import dearpygui.dearpygui as dpg
from typing import Dict, List, Callable, Optional
from desktop_ui.theme import create_panel_theme, Colors, rgb_to_dpg


class PipelinePanel:
    """
    Pipeline konfiguracios panel a 3 feldolgozasi lepcsohz.
    """

    def __init__(self):
        self.theme = None
        self.preprocessors: List[str] = []
        self.feature_extractors: List[str] = []
        self.classifiers: List[str] = []
        self.game_modes: List[str] = ["Single Hand", "Two Player Game"]

        self.on_change_callback: Optional[Callable] = None

        # Komponens ID-k
        self.preproc_combo = None
        self.feature_combo = None
        self.classifier_combo = None
        self.game_mode_combo = None

    def set_options(self, preprocessors: List[str], feature_extractors: List[str],
                    classifiers: List[str]):
        """Beallitja a valaszthato opcokat."""
        self.preprocessors = preprocessors
        self.feature_extractors = feature_extractors
        self.classifiers = classifiers

    def set_callback(self, callback: Callable):
        """Callback amikor valtozik valamelyik opcio."""
        self.on_change_callback = callback

    def _on_combo_change(self, sender, app_data):
        """Belso callback combo valtozaskor."""
        if self.on_change_callback:
            self.on_change_callback()

    def create(self, parent: int, width: int):
        """Pipeline panel letrehozasa."""
        self.theme = create_panel_theme()

        with dpg.child_window(parent=parent, width=width, height=180,
                              border=True, tag="pipeline_panel"):
            dpg.bind_item_theme("pipeline_panel", self.theme)

            dpg.add_spacer(height=5)

            # Game Mode valaszto
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("  Mode", color=rgb_to_dpg(Colors.TEXT_SECONDARY))
                dpg.add_spacer(width=20)
                self.game_mode_combo = dpg.add_combo(
                    items=self.game_modes,
                    default_value=self.game_modes[0],
                    width=200,
                    callback=self._on_combo_change
                )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Pipeline lepcsok egy sorban
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)

                # Step 1: Preprocessing
                with dpg.group():
                    with dpg.group(horizontal=True):
                        self._create_step_badge(1)
                        dpg.add_text("Preprocessing", color=rgb_to_dpg(Colors.TEXT_PRIMARY))
                    dpg.add_spacer(height=5)
                    self.preproc_combo = dpg.add_combo(
                        items=self.preprocessors if self.preprocessors else ["None"],
                        default_value=self.preprocessors[0] if self.preprocessors else "None",
                        width=160,
                        callback=self._on_combo_change
                    )

                # Nyil
                dpg.add_spacer(width=10)
                dpg.add_text("->", color=rgb_to_dpg(Colors.PRIMARY))
                dpg.add_spacer(width=10)

                # Step 2: Feature Extraction
                with dpg.group():
                    with dpg.group(horizontal=True):
                        self._create_step_badge(2)
                        dpg.add_text("Feature Extraction", color=rgb_to_dpg(Colors.TEXT_PRIMARY))
                    dpg.add_spacer(height=5)
                    self.feature_combo = dpg.add_combo(
                        items=self.feature_extractors if self.feature_extractors else ["None"],
                        default_value=self.feature_extractors[0] if self.feature_extractors else "None",
                        width=180,
                        callback=self._on_combo_change
                    )

                # Nyil
                dpg.add_spacer(width=10)
                dpg.add_text("->", color=rgb_to_dpg(Colors.PRIMARY))
                dpg.add_spacer(width=10)

                # Step 3: Classification
                with dpg.group():
                    with dpg.group(horizontal=True):
                        self._create_step_badge(3)
                        dpg.add_text("Classification", color=rgb_to_dpg(Colors.TEXT_PRIMARY))
                    dpg.add_spacer(height=5)
                    self.classifier_combo = dpg.add_combo(
                        items=self.classifiers if self.classifiers else ["None"],
                        default_value=self.classifiers[0] if self.classifiers else "None",
                        width=160,
                        callback=self._on_combo_change
                    )

    def _create_step_badge(self, number: int):
        """Letrehoz egy szamozott badge-et."""
        dpg.add_text(f"[{number}]", color=rgb_to_dpg(Colors.PRIMARY))

    def get_preprocessing(self) -> str:
        """Aktualis preprocessing valasztas."""
        if self.preproc_combo:
            return dpg.get_value(self.preproc_combo)
        return "None"

    def get_feature_extractor(self) -> str:
        """Aktualis feature extractor valasztas."""
        if self.feature_combo:
            return dpg.get_value(self.feature_combo)
        return ""

    def get_classifier(self) -> str:
        """Aktualis classifier valasztas."""
        if self.classifier_combo:
            return dpg.get_value(self.classifier_combo)
        return ""

    def get_game_mode(self) -> str:
        """Aktualis game mode valasztas."""
        if self.game_mode_combo:
            return dpg.get_value(self.game_mode_combo)
        return "Single Hand"
