"""
Header panel komponens.
"""

import dearpygui.dearpygui as dpg
from desktop_ui.theme import create_header_theme, Colors, rgb_to_dpg


class HeaderPanel:
    """
    Alkalmazas fejlec panel.
    """

    def __init__(self, title: str = "Rock-Paper-Scissors",
                 subtitle: str = "Hand gesture recognition with computer vision"):
        self.title = title
        self.subtitle = subtitle
        self.theme = None

    def create(self, parent: int, width: int):
        """Header letrehozasa."""
        self.theme = create_header_theme()

        with dpg.child_window(parent=parent, width=width, height=80,
                              border=False, tag="header_panel"):
            dpg.bind_item_theme("header_panel", self.theme)

            dpg.add_spacer(height=8)

            # Fo cim
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=15)
                dpg.add_text(f"  {self.title}", color=rgb_to_dpg(Colors.TEXT_PRIMARY))

            # Alcim
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=15)
                dpg.add_text(f"     {self.subtitle}",
                           color=rgb_to_dpg((220, 220, 255)))
