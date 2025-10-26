"""
Gradio fő interface 3-stage pipeline támogatással.
UI komponensek a ui.components modulból töltődnek
"""

import gradio as gr
import numpy as np
from typing import Dict, Optional

from ui.styles import get_custom_css
from ui.components import (
    create_header,
    create_pipeline_arrow,
    create_step_1,
    create_step_2,
    create_step_3,
    ResultStep,
    InputSection,
    ControlsSection,
    PreviewSection,
    PipelineInfo,
    DebugLog
)


class MainInterface:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.preprocessors: Dict[str, Optional[object]] = {}
        self.feature_extractors: Dict[str, Optional[object]] = {}
        self.classifiers: Dict[str, Optional[object]] = {}
        self.current_image = None
        self.processed_image = None
        self.current_features = None

        self.input_section = InputSection()
        self.controls_section = ControlsSection()
        self.preview_section = PreviewSection()
        self.result_step = ResultStep()
        self.pipeline_info = PipelineInfo()
        self.debug_log = DebugLog()

    def register_preprocessor(self, name: str, module):
        self.preprocessors[name] = module

    def register_feature_extractor(self, name: str, extractor):
        self.feature_extractors[name] = extractor

    def register_classifier(self, name: str, classifier):
        self.classifiers[name] = classifier

    def process_image(self, image, preprocessing_name: str,
                      feature_extractor_name: str, classifier_name: str,
                      display_mode: str):
        if image is None:
            return (
                None,
                *self.result_step.reset(),
                self.pipeline_info.update("Preprocessing: No preprocessing"),
                self.debug_log.set_text("[Info] No image to process")
            )

        self.current_image = image
        self.debug_log.clear()

        self.pipeline.clear_preprocessing()
        if preprocessing_name in self.preprocessors and self.preprocessors[preprocessing_name] is not None:
            self.pipeline.add_preprocessing(self.preprocessors[preprocessing_name])
            self.debug_log.append(f"[Stage 1] Preprocessing: {preprocessing_name}")
        else:
            self.debug_log.append(f"[Stage 1] Preprocessing: None")

        if feature_extractor_name not in self.feature_extractors:
            self.debug_log.append("[Stage 2] ERROR: No feature extractor selected")
            return (
                image,
                *self.result_step.set_error(),
                self.pipeline_info.update("Pipeline incomplete"),
                self.debug_log.get_text()
            )

        self.pipeline.set_feature_extractor(self.feature_extractors[feature_extractor_name])
        self.debug_log.append(f"[Stage 2] Feature Extractor: {feature_extractor_name}")

        if classifier_name not in self.classifiers:
            self.debug_log.append("[Stage 3] ERROR: No classifier selected")
            return (
                image,
                *self.result_step.set_error(),
                self.pipeline_info.update("Pipeline incomplete"),
                self.debug_log.get_text()
            )

        self.pipeline.set_classifier(self.classifiers[classifier_name])
        self.debug_log.append(f"[Stage 3] Classifier: {classifier_name}")

        try:
            self.debug_log.append("\n[Pipeline] Starting full processing...")

            preprocessed, features, result = self.pipeline.process_full_pipeline(image)

            # ==================================================
            # átmeneti megoldás: ha lista a pipeline output, akkor csak az 1. elemekkel foglalkozunk

            if isinstance(preprocessed, list):
                self.debug_log.append(f"preproc count: {len(preprocessed)}")
                preprocessed=preprocessed[0].data
            else:
                preprocessed=preprocessed.data
            if isinstance(features, list):
                self.debug_log.append(f"features count: {len(features)}")
                features=features[0]
            if isinstance(result, list):
                self.debug_log.append(f"classifications count: {len(result)}")
                result=result[0]
            
            # átmeneti megoldás vége
            # ==================================================

            self.processed_image = preprocessed
            self.current_features = features

            self.debug_log.append(f"\n[Features] Type: {features.feature_type.value}")
            self.debug_log.append(f"[Features] Dimension: {features.feature_dimension}")
            self.debug_log.append(f"[Features] Time: {features.extraction_time_ms:.2f}ms")

            if features.named_features:
                self.debug_log.append("[Features] Values:")
                for key, value in features.named_features.items():
                    self.debug_log.append(f"  - {key}: {value:.4f}")

            self.debug_log.append(f"\n[Classification] Predicted: {result.predicted_class.value.upper()}")
            self.debug_log.append(f"[Classification] Confidence: {result.get_confidence_percentage():.1f}%")
            self.debug_log.append(f"[Classification] Time: {result.processing_time_ms:.2f}ms")

            class_text, conf_text = self.result_step.update(
                result.predicted_class.value,
                result.get_confidence_percentage()
            )

            self.debug_log.append("\n[Pipeline] ✓ Complete!")

        except Exception as e:
            self.debug_log.append(f"\n[ERROR] {str(e)}")
            class_text, conf_text = self.result_step.set_error()
            preprocessed = image

        if display_mode == "Original":
            output_image = self.current_image
        else:
            output_image = self.processed_image if self.processed_image is not None else self.current_image

        self.preview_section.set_images(self.current_image, self.processed_image)
        self.preview_section.set_display_mode(display_mode)

        pipeline_info_text = self.pipeline_info.update(self.pipeline.get_pipeline_info())

        return (
            output_image,
            class_text,
            conf_text,
            pipeline_info_text,
            self.debug_log.get_text()
        )

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            css=get_custom_css(),
            title="Rock-Paper-Scissors",
            theme=gr.themes.Soft()
        ) as demo:

            create_header()

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        preproc_dropdown = create_step_1(self.preprocessors)
                        create_pipeline_arrow()
                        feature_dropdown = create_step_2(self.feature_extractors)
                        create_pipeline_arrow()
                        classifier_dropdown = create_step_3(self.classifiers)
                        create_pipeline_arrow()

                    with gr.Row():
                        input_image = self.input_section.create()
                        process_btn = self.controls_section.create()

                with gr.Column(scale=1):
                    class_out, conf_out = self.result_step.create()
                    pipeline_out = self.pipeline_info.create()
                    log_out = self.debug_log.create()

            with gr.Row():
                output_image, display_radio = self.preview_section.create()

            inputs = [
                input_image,
                preproc_dropdown,
                feature_dropdown,
                classifier_dropdown,
                display_radio
            ]
            outputs = [output_image, class_out, conf_out, pipeline_out, log_out]

            input_image.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            process_btn.click(fn=self.process_image, inputs=inputs, outputs=outputs)
            preproc_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            feature_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            classifier_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            display_radio.change(fn=self.process_image, inputs=inputs, outputs=outputs)

        return demo