"""
Image input és controls komponensek.
"""

import gradio as gr
import numpy as np
from typing import Optional
from ui.utils import render


class InputSection:
    def __init__(self):
        self.image_component = None
        self.current_image: Optional[np.ndarray] = None
        self.image_history = []

    def create(self):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-controls-section"):
                gr.HTML(render('section_label.html', icon="📤", title="Image Input"))
                self.image_component = gr.Image(
                    label="",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=70,
                    container=False
                )
        return self.image_component

    def set_image(self, image: Optional[np.ndarray]):
        if image is not None:
            self.current_image = image
            self.image_history.append(image)
            if len(self.image_history) > 10:
                self.image_history.pop(0)

    def get_image(self) -> Optional[np.ndarray]:
        return self.current_image

    def clear(self):
        self.current_image = None
        return None

    def get_history(self):
        return self.image_history.copy()

    def clear_history(self):
        self.image_history = []


class ControlsSection:
    def __init__(self):
        self.button_component = None
        self.is_processing = False

    def create(self):
        with gr.Column(scale=2):
            with gr.Group(elem_classes="input-controls-section"):
                gr.HTML(render('section_label.html', icon="🎛️", title="Controls"))
                self.button_component = gr.Button(
                    "▶ Process Image",
                    size="lg",
                    elem_classes="process-btn"
                )
        return self.button_component

    def set_processing(self, state: bool):
        self.is_processing = state

    def is_busy(self) -> bool:
        return self.is_processing
