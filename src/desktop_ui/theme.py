"""
Dear PyGui theme - a Gradio UI szineit es stilusat koveti.
"""

import dearpygui.dearpygui as dpg


# Szinek (RGB 0-255)
class Colors:
    # Gradient szinek (header, kiemelt elemek)
    PRIMARY = (102, 126, 234)        # #667eea
    SECONDARY = (118, 75, 162)       # #764ba2

    # Gomb szinek
    BUTTON_GREEN = (72, 187, 120)    # #48bb78
    BUTTON_GREEN_HOVER = (56, 161, 105)  # #38a169

    # Hatter szinek
    BG_DARK = (30, 30, 30)           # #1e1e1e
    BG_MEDIUM = (45, 45, 48)         # #2d2d30
    BG_LIGHT = (60, 60, 64)          # #3c3c40

    # Szoveg szinek
    TEXT_PRIMARY = (255, 255, 255)   # Feher
    TEXT_SECONDARY = (180, 180, 180) # Halvany
    TEXT_MUTED = (128, 128, 128)     # Muted

    # Border szinek
    BORDER = (80, 80, 85)            # Keret
    BORDER_ACTIVE = (102, 126, 234)  # Aktiv keret (PRIMARY)

    # Allapot szinek
    SUCCESS = (72, 187, 120)         # Zold
    ERROR = (239, 68, 68)            # Piros
    WARNING = (245, 158, 11)         # Sarga


def rgb_to_dpg(rgb, alpha=255):
    """RGB tuple (0-255) konvertalasa Dear PyGui formatumra."""
    return (rgb[0], rgb[1], rgb[2], alpha)


def create_theme():
    """Letrehozza a globalis Dear PyGui theme-et."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Hatter szinek
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, rgb_to_dpg(Colors.BG_DARK))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.BG_MEDIUM))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, rgb_to_dpg(Colors.BG_MEDIUM))

            # Frame (input mezok) szinek
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, rgb_to_dpg(Colors.BG_LIGHT))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, rgb_to_dpg(Colors.BG_LIGHT, 200))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, rgb_to_dpg(Colors.PRIMARY, 100))

            # Gomb szinek
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgb_to_dpg(Colors.PRIMARY))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgb_to_dpg(Colors.SECONDARY))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgb_to_dpg(Colors.PRIMARY))

            # Header szinek
            dpg.add_theme_color(dpg.mvThemeCol_Header, rgb_to_dpg(Colors.PRIMARY, 100))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, rgb_to_dpg(Colors.PRIMARY, 150))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, rgb_to_dpg(Colors.PRIMARY))

            # Border
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgb_to_dpg(Colors.BORDER))

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgb_to_dpg(Colors.TEXT_PRIMARY))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, rgb_to_dpg(Colors.TEXT_MUTED))

            # Title
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, rgb_to_dpg(Colors.BG_DARK))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, rgb_to_dpg(Colors.PRIMARY))

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, rgb_to_dpg(Colors.BG_DARK))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, rgb_to_dpg(Colors.BG_LIGHT))

            # Tab
            dpg.add_theme_color(dpg.mvThemeCol_Tab, rgb_to_dpg(Colors.BG_MEDIUM))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, rgb_to_dpg(Colors.PRIMARY))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, rgb_to_dpg(Colors.PRIMARY))

            # Stilusok
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)

            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)

    return global_theme


def create_button_green_theme():
    """Zold gomb theme (Process gomb)."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgb_to_dpg(Colors.BUTTON_GREEN))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgb_to_dpg(Colors.BUTTON_GREEN_HOVER))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgb_to_dpg(Colors.BUTTON_GREEN))
    return theme


def create_header_theme():
    """Header panel theme gradient hatassal."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.PRIMARY))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
    return theme


def create_panel_theme():
    """Standard panel theme kerettel."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.BG_MEDIUM))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgb_to_dpg(Colors.BORDER))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 2)
    return theme


def create_result_panel_theme():
    """Eredmeny panel theme kiemelt kerettel."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.BG_MEDIUM))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgb_to_dpg(Colors.PRIMARY))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 2)
    return theme


def create_preview_theme():
    """Preview panel theme kiemelt kerettel."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgb_to_dpg(Colors.BG_MEDIUM))
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgb_to_dpg(Colors.PRIMARY))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 3)
    return theme
