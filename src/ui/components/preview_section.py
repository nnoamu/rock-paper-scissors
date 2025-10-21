"""
Image preview komponens display mode választóval.

PreviewSection class-alapú a display mode és image cache kezeléshez.
"""

import gradio as gr
import numpy as np
from typing import Optional, Literal
from ui.utils import render

DisplayMode = Literal["Original", "Preprocessed"]


class PreviewSection:
    def __init__(self):
        self.image_component = None
        self.radio_component = None
        self.display_mode: DisplayMode = "Original"
        self.original_image: Optional[np.ndarray] = None
        self.preprocessed_image: Optional[np.ndarray] = None

    def create(self):
        with gr.Column():
            with gr.Group(elem_classes="preview-section"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(render('section_label.html', icon="🖼️", title="Image Preview"))
                    with gr.Column(scale=1):
                        self.radio_component = gr.Radio(
                            choices=["Original", "Preprocessed"],
                            value=self.display_mode,
                            label="",
                            container=False,
                            show_label=False
                        )

                self.image_component = gr.Image(
                    label="",
                    type="numpy",
                    height=500,
                    container=False,
                    interactive=False
                )

        return self.image_component, self.radio_component

    def set_images(self, original: Optional[np.ndarray], preprocessed: Optional[np.ndarray]):
        self.original_image = original
        self.preprocessed_image = preprocessed

    def set_display_mode(self, mode: DisplayMode):
        self.display_mode = mode

    def get_current_image(self) -> Optional[np.ndarray]:
        if self.display_mode == "Original":
            return self.original_image
        else:
            return self.preprocessed_image if self.preprocessed_image is not None else self.original_image

    def update_display(self, mode: DisplayMode) -> Optional[np.ndarray]:
        self.set_display_mode(mode)
        return self.get_current_image()

    def clear(self):
        self.original_image = None
        self.preprocessed_image = None
        return None
