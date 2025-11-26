"""
Eredmeny megjelenitesi panel.
"""

import dearpygui.dearpygui as dpg
from desktop_ui.theme import create_result_panel_theme, Colors, rgb_to_dpg


class ResultPanel:
    """
    Osztalyozasi eredmeny megjelenitese.
    """

    def __init__(self):
        self.theme = None
        self.class_text = None
        self.confidence_text = None
        self.class_bg = None

    def create(self, parent: int, width: int):
        """Result panel letrehozasa."""
        self.theme = create_result_panel_theme()

        with dpg.child_window(parent=parent, width=width, height=140,
                              border=True, tag="result_panel"):
            dpg.bind_item_theme("result_panel", self.theme)

            dpg.add_spacer(height=5)

            # Header
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("  RESULT", color=rgb_to_dpg(Colors.PRIMARY))

            dpg.add_spacer(height=10)

            # Osztalyozas eredmenye (nagy, kozepen)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                # Hatter "gomb" az eredmenyhez
                with dpg.child_window(width=width-40, height=40, border=False,
                                      tag="class_bg_panel"):
                    with dpg.theme() as bg_theme:
                        with dpg.theme_component(dpg.mvChildWindow):
                            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.PRIMARY))
                            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                    dpg.bind_item_theme("class_bg_panel", bg_theme)

                    dpg.add_spacer(height=8)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=10)
                        self.class_text = dpg.add_text("  -",
                                                        color=rgb_to_dpg(Colors.TEXT_PRIMARY))

            dpg.add_spacer(height=10)

            # Confidence
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("Confidence:", color=rgb_to_dpg(Colors.TEXT_SECONDARY))
                dpg.add_spacer(width=10)
                self.confidence_text = dpg.add_text("- %",
                                                     color=rgb_to_dpg(Colors.TEXT_PRIMARY))

    def update(self, predicted_class: str, confidence: float):
        """Eredmeny frissitese."""
        if self.class_text:
            dpg.set_value(self.class_text, f"  {predicted_class.upper()}")
        if self.confidence_text:
            dpg.set_value(self.confidence_text, f"{confidence:.1f}%")

    def update_game_result(self, result_text: str, details: str):
        """Jatek eredmeny frissitese (Two Player mode)."""
        if self.class_text:
            dpg.set_value(self.class_text, result_text)
        if self.confidence_text:
            dpg.set_value(self.confidence_text, details)

    def set_error(self):
        """Hiba allapot."""
        if self.class_text:
            dpg.set_value(self.class_text, "  ERROR")
        if self.confidence_text:
            dpg.set_value(self.confidence_text, "-")

    def reset(self):
        """Alapallapot."""
        if self.class_text:
            dpg.set_value(self.class_text, "  -")
        if self.confidence_text:
            dpg.set_value(self.confidence_text, "- %")
