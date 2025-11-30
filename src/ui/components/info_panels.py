"""
Pipeline Info és Debug Log panelek class-alapú implementációval.

Példa használat:
    debug_log = DebugLog()
    log_component = debug_log.create()

    # Később írni bele:
    debug_log.append("[INFO] Pipeline started")
    debug_log.clear()
"""

import gradio as gr
from ui.utils import render


class PipelineInfo:
    def __init__(self):
        self.component = None
        self._current_text = "Preprocessing: No preprocessing"

    def create(self):
        with gr.Group(elem_classes="info-section"):
            gr.HTML(render('info_title.html', icon="🔧", title="Pipeline Info"))
            self.component = gr.Textbox(
                label="",
                value=self._current_text,
                container=False,
                interactive=False,
                lines=3
            )
        return self.component

    def update(self, text: str):
        self._current_text = text
        return text


class DebugLog:
    def __init__(self):
        self.component = None
        self._log_lines = []

    def create(self):
        with gr.Group(elem_classes="info-section"):
            gr.HTML(render('info_title.html', icon="📋", title="Debug Log"))
            self.component = gr.Textbox(
                label="",
                value="",
                container=False,
                interactive=False,
                lines=20,
                elem_classes="debug-log"
            )
        return self.component

    def append(self, message: str):
        self._log_lines.append(message)
        return "\n".join(self._log_lines)

    def clear(self):
        self._log_lines = []
        return ""

    def get_text(self):
        return "\n".join(self._log_lines)

    def set_text(self, text: str):
        self._log_lines = text.split("\n") if text else []
        return text


def create_pipeline_info():
    info = PipelineInfo()
    return info.create()


def create_debug_log():
    log = DebugLog()
    return log.create()
