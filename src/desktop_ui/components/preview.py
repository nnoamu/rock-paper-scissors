"""
Kep preview panel komponens.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, Literal
from desktop_ui.theme import create_preview_theme, Colors, rgb_to_dpg
from desktop_ui.utils.image_utils import numpy_to_dpg_texture, resize_with_aspect, create_placeholder_image

DisplayMode = Literal["Original", "Preprocessed", "Annotated"]


class PreviewPanel:
    """
    Kep preview panel display mode valasztoval.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.theme = None

        self.texture_tag = None
        self.image_tag = None
        self.radio_group = None

        self.display_mode: DisplayMode = "Annotated"
        self.original_image: Optional[np.ndarray] = None
        self.preprocessed_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None

        self._texture_created = False

    def create(self, parent: int, panel_width: int):
        """Preview panel letrehozasa."""
        self.theme = create_preview_theme()

        # Texture registry letrehozasa
        with dpg.texture_registry(show=False):
            placeholder = create_placeholder_image(self.width, self.height, "No Image")
            placeholder_data = numpy_to_dpg_texture(placeholder)
            self.texture_tag = dpg.add_raw_texture(
                width=self.width,
                height=self.height,
                default_value=placeholder_data,
                format=dpg.mvFormat_Float_rgba,
                tag="preview_texture"
            )
            self._texture_created = True

        with dpg.child_window(parent=parent, width=panel_width, height=self.height + 80,
                              border=True, tag="preview_panel"):
            dpg.bind_item_theme("preview_panel", self.theme)

            # Header + Radio
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("  IMAGE PREVIEW", color=rgb_to_dpg(Colors.PRIMARY))
                dpg.add_spacer(width=50)

                # Display mode radio
                dpg.add_text("Display:", color=rgb_to_dpg(Colors.TEXT_SECONDARY))
                dpg.add_spacer(width=10)
                self.radio_group = dpg.add_radio_button(
                    items=["Original", "Preprocessed", "Annotated"],
                    default_value="Annotated",
                    horizontal=True,
                    callback=self._on_mode_change
                )

            dpg.add_spacer(height=10)

            # Kep megjelenites
            with dpg.group(horizontal=True):
                # Kozepre igazitas
                dpg.add_spacer(width=(panel_width - self.width) // 2 - 20)
                self.image_tag = dpg.add_image("preview_texture")

    def _on_mode_change(self, sender, app_data):
        """Display mode valtaskor."""
        self.display_mode = app_data
        self._update_display()

    def set_images(self, original: Optional[np.ndarray],
                   preprocessed: Optional[np.ndarray],
                   annotated: Optional[np.ndarray] = None):
        """Kepek beallitasa."""
        self.original_image = original
        self.preprocessed_image = preprocessed
        self.annotated_image = annotated
        self._update_display()

    def _update_display(self):
        """Kep frissitese az aktualis display mode szerint."""
        image = self._get_current_image()

        if image is None:
            image = create_placeholder_image(self.width, self.height, "No Image")
        else:
            # Meretezd at ha kell
            image = resize_with_aspect(image, self.width, self.height)
            # Padding hogy pont a meret legyen
            h, w = image.shape[:2]
            if h != self.height or w != self.width:
                padded = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                padded[:] = (45, 45, 48)
                y_off = (self.height - h) // 2
                x_off = (self.width - w) // 2
                padded[y_off:y_off+h, x_off:x_off+w] = image if len(image.shape) == 3 else np.stack([image]*3, axis=-1)
                image = padded

        self._update_texture(image)

    def _get_current_image(self) -> Optional[np.ndarray]:
        """Aktualis kep a display mode szerint."""
        if self.display_mode == "Original":
            return self.original_image
        elif self.display_mode == "Annotated":
            return self.annotated_image if self.annotated_image is not None else self.original_image
        else:  # Preprocessed
            return self.preprocessed_image if self.preprocessed_image is not None else self.original_image

    def _update_texture(self, image: np.ndarray):
        """Texture frissitese uj keppel."""
        if not self._texture_created:
            return

        # Biztositsd a helyes meretet
        h, w = image.shape[:2]
        if h != self.height or w != self.width:
            image = resize_with_aspect(image, self.width, self.height)

        texture_data = numpy_to_dpg_texture(image)
        dpg.set_value("preview_texture", texture_data)

    def update_live(self, frame: np.ndarray):
        """Elo kamerakep frissitese (eredeti kepkent)."""
        self.original_image = frame
        if self.display_mode == "Original":
            self._update_display()

    def clear(self):
        """Kepek torlese."""
        self.original_image = None
        self.preprocessed_image = None
        self.annotated_image = None
        self._update_display()
