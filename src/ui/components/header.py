"""
Header komponens
"""

import gradio as gr
from ui.utils import render


def create_header(
        title="Rock-Paper-Scissors",
        subtitle="Upload an image or use your webcam to detect hand gestures",
        icon="🎮"
):
    with gr.Row():
        gr.HTML(render('header.html', title=title, subtitle=subtitle, icon=icon))
