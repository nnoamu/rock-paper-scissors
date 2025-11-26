"""
Debug log panel komponens.
"""

import dearpygui.dearpygui as dpg
from typing import List
from desktop_ui.theme import create_panel_theme, Colors, rgb_to_dpg


class DebugLogPanel:
    """
    Debug log megjelenito panel.
    """

    def __init__(self):
        self.theme = None
        self.log_text = None
        self._log_lines: List[str] = []
        self._max_lines = 100

    def create(self, parent: int, width: int, height: int = 200):
        """Debug log panel letrehozasa."""
        self.theme = create_panel_theme()

        with dpg.child_window(parent=parent, width=width, height=height,
                              border=True, tag="debug_panel"):
            dpg.bind_item_theme("debug_panel", self.theme)

            # Header
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("  DEBUG LOG", color=rgb_to_dpg(Colors.PRIMARY))

            dpg.add_spacer(height=5)

            # Log szoveg (multiline, monospace stilusu)
            self.log_text = dpg.add_input_text(
                multiline=True,
                readonly=True,
                width=width - 30,
                height=height - 50,
                default_value="",
                tab_input=False
            )

            # Sotetkek hatter a loghoz
            with dpg.theme() as log_theme:
                with dpg.theme_component(dpg.mvInputText):
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, rgb_to_dpg((30, 30, 30)))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, rgb_to_dpg((200, 200, 200)))
            dpg.bind_item_theme(self.log_text, log_theme)

    def append(self, message: str):
        """Uj log sor hozzaadasa."""
        self._log_lines.append(message)

        # Max sorok korlatozasa
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines:]

        self._update_text()

    def clear(self):
        """Log torlese."""
        self._log_lines = []
        self._update_text()

    def _update_text(self):
        """Szoveg frissitese a widgetben."""
        if self.log_text:
            text = "\n".join(self._log_lines)
            dpg.set_value(self.log_text, text)

    def get_text(self) -> str:
        """Teljes log szoveg."""
        return "\n".join(self._log_lines)

    def set_text(self, text: str):
        """Log szoveg beallitasa."""
        self._log_lines = text.split("\n") if text else []
        self._update_text()


class PipelineInfoPanel:
    """
    Pipeline info panel (roviditett info a pipelinerol).
    """

    def __init__(self):
        self.theme = None
        self.info_text = None

    def create(self, parent: int, width: int, height: int = 100):
        """Info panel letrehozasa."""
        self.theme = create_panel_theme()

        with dpg.child_window(parent=parent, width=width, height=height,
                              border=True, tag="pipeline_info_panel"):
            dpg.bind_item_theme("pipeline_info_panel", self.theme)

            # Header
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("  PIPELINE INFO", color=rgb_to_dpg(Colors.PRIMARY))

            dpg.add_spacer(height=5)

            # Info szoveg
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                self.info_text = dpg.add_text(
                    "Preprocessing: None",
                    color=rgb_to_dpg(Colors.TEXT_SECONDARY),
                    wrap=width - 30
                )

    def update(self, info: str):
        """Info szoveg frissitese."""
        if self.info_text:
            dpg.set_value(self.info_text, info)
