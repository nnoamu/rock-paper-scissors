"""
Pipeline lépések komponensei (Step 1, 2, 3, Result).
"""

import gradio as gr
from ui.utils import render


def create_pipeline_arrow():
    with gr.Column(scale=0, min_width=50):
        gr.HTML(render('pipeline_arrow.html'))


def create_step_1(preprocessors):
    with gr.Column(scale=1):
        with gr.Group(elem_classes="pipeline-step-small active"):
            gr.HTML(render(
                'step_header.html',
                number=1,
                title="Preprocessing",
                disabled=False
            ))
            with gr.Group(elem_classes="step-content"):
                preproc_dropdown = gr.Dropdown(
                    choices=list(preprocessors.keys()),
                    value=list(preprocessors.keys())[0] if preprocessors else "None",
                    label="",
                    container=False
                )
    return preproc_dropdown


def create_step_2(feature_extractors):
    with gr.Column(scale=1):
        with gr.Group(elem_classes="pipeline-step-small active"):
            gr.HTML(render(
                'step_header.html',
                number=2,
                title="Feature Extraction",
                disabled=False
            ))
            with gr.Group(elem_classes="step-content"):
                feature_dropdown = gr.Dropdown(
                    choices=list(feature_extractors.keys()),
                    value=list(feature_extractors.keys())[0] if feature_extractors else None,
                    label="",
                    container=False
                )
    return feature_dropdown


def create_step_3(classifiers):
    with gr.Column(scale=1):
        with gr.Group(elem_classes="pipeline-step-small active"):
            gr.HTML(render(
                'step_header.html',
                number=3,
                title="Classification",
                disabled=False
            ))
            with gr.Group(elem_classes="step-content"):
                classifier_dropdown = gr.Dropdown(
                    choices=list(classifiers.keys()),
                    value=list(classifiers.keys())[0] if classifiers else None,
                    label="",
                    container=False
                )
    return classifier_dropdown


def create_pipeline_steps_block(preprocessors, feature_extractors, classifiers):

    with gr.Group():
        with gr.Row():
            preproc_dropdown = create_step_1(preprocessors)
        with gr.Row():
            feature_dropdown = create_step_2(feature_extractors)
        with gr.Row():
            classifier_dropdown = create_step_3(classifiers)
    
    return preproc_dropdown, feature_dropdown, classifier_dropdown


class ResultStep:

    def __init__(self):
        self.class_component = None
        self.confidence_component = None
        self._current_class = "🎯 -"
        self._current_confidence = "📊 -%"

    def create(self):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="pipeline-step-result"):
                gr.HTML(render('result_header.html'))
                with gr.Group(elem_classes="step-content"):
                    self.class_component = gr.Textbox(
                        label="",
                        value=self._current_class,
                        container=False,
                        interactive=False,
                        elem_classes="class-output-inline"
                    )
                    self.confidence_component = gr.Textbox(
                        label="",
                        value=self._current_confidence,
                        container=False,
                        interactive=False,
                        elem_classes="confidence-output-inline"
                    )
        return self.class_component, self.confidence_component

    def create_display_only(self):
        """Create only the display components without the header, for use under image preview."""
        self.class_component = gr.Textbox(
            label="",
            value=self._current_class,
            container=False,
            interactive=False,
            elem_classes="class-output-inline"
        )
        self.confidence_component = gr.Textbox(
            label="",
            value=self._current_confidence,
            container=False,
            interactive=False,
            elem_classes="confidence-output-inline"
        )
        return self.class_component, self.confidence_component

    def update(self, predicted_class: str, confidence: float):
        self._current_class = f"🎯 {predicted_class.upper()}"
        self._current_confidence = f"📊 {confidence:.1f}%"
        return self._current_class, self._current_confidence

    def set_error(self):
        self._current_class = "🎯 ERROR"
        self._current_confidence = "📊 -"
        return self._current_class, self._current_confidence

    def reset(self):
        self._current_class = "🎯 -"
        self._current_confidence = "📊 -%"
        return self._current_class, self._current_confidence
