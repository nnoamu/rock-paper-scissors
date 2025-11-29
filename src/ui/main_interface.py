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
    create_pipeline_steps_block,
    ResultStep,
    InputSection,
    ControlsSection,
    PreviewSection,
    PipelineInfo,
    DebugLog
)
from game import TwoPlayerGameWrapper


class MainInterface:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.game_wrapper: Optional[TwoPlayerGameWrapper] = None
        self.preprocessors: Dict[str, Optional[object]] = {}
        self.feature_extractors: Dict[str, Optional[object]] = {}
        self.classifiers: Dict[str, Optional[object]] = {}
        self.current_image = None
        self.processed_image = None
        self.annotated_image = None
        self.current_features = None

        self.input_section = InputSection()
        self.controls_section = ControlsSection()
        self.preview_section = PreviewSection()
        self.result_step = ResultStep()
        self.pipeline_info = PipelineInfo()
        self.debug_log = DebugLog()

    def set_game_wrapper(self, game_wrapper: TwoPlayerGameWrapper):
        self.game_wrapper = game_wrapper

    def register_preprocessor(self, name: str, module):
        self.preprocessors[name] = module

    def register_feature_extractor(self, name: str, extractor):
        self.feature_extractors[name] = extractor

    def register_classifier(self, name: str, classifier):
        self.classifiers[name] = classifier

    def process_image(self, image, preprocessing_name: str,
                      feature_extractor_name: str, classifier_name: str,
                      display_mode: str, game_mode: str = "Single Hand"):
        if image is None:
            return (
                None,
                *self.result_step.reset(),
                self.pipeline_info.update("Preprocessing: No preprocessing"),
                self.debug_log.set_text("[Info] No image to process")
            )

        if game_mode == "Two Player Game":
            return self._process_game_mode(image, preprocessing_name,
                                          feature_extractor_name, classifier_name,
                                          display_mode)

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

            preprocessed, features, result, annotated = self.pipeline.process_full_pipeline(image)

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
            self.annotated_image = annotated
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
        elif display_mode == "Annotated":
            output_image = self.annotated_image if self.annotated_image is not None else self.current_image
        else:  # Preprocessed
            output_image = self.processed_image if self.processed_image is not None else self.current_image

        self.preview_section.set_images(self.current_image, self.processed_image, self.annotated_image)
        self.preview_section.set_display_mode(display_mode)

        pipeline_info_text = self.pipeline_info.update(self.pipeline.get_pipeline_info())

        return (
            output_image,
            class_text,
            conf_text,
            pipeline_info_text,
            self.debug_log.get_text()
        )

    def _process_game_mode(self, image, preprocessing_name: str,
                           feature_extractor_name: str, classifier_name: str,
                           display_mode: str):
        """Process image in two-player game mode."""
        self.current_image = image
        self.debug_log.clear()

        # Check if game wrapper is available
        if self.game_wrapper is None:
            self.debug_log.append("[ERROR] Game mode not available - game wrapper not initialized")
            return (
                image,
                *self.result_step.set_error(),
                self.pipeline_info.update("Game mode error"),
                self.debug_log.get_text()
            )

        # Setup pipeline for game mode
        self.pipeline.clear_preprocessing()

        from preprocessing.image_splitter import ImageSplitterModule
        self.pipeline.add_preprocessing(ImageSplitterModule())
        self.debug_log.append(f"[Stage 1a] Image Splitter: Split image for 2-player batch mode")

        if preprocessing_name in self.preprocessors and self.preprocessors[preprocessing_name] is not None:
            self.pipeline.add_preprocessing(self.preprocessors[preprocessing_name])
            self.debug_log.append(f"[Stage 1b] Additional Preprocessing: {preprocessing_name}")
        else:
            self.debug_log.append(f"[Stage 1b] Additional Preprocessing: None")

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
            self.debug_log.append("\n[Game Mode] Running two-player game evaluation...")

            # Run game wrapper with visualization
            game_result, annotated = self.game_wrapper.run_with_visualization(image)

            # Store annotated image
            self.annotated_image = annotated

            # Log player results
            p1 = game_result.player1_result
            p2 = game_result.player2_result

            self.debug_log.append(f"\n[Player 1] Class: {p1.predicted_class.value.upper()}")
            self.debug_log.append(f"[Player 1] Confidence: {p1.get_confidence_percentage():.1f}%")
            self.debug_log.append(f"\n[Player 2] Class: {p2.predicted_class.value.upper()}")
            self.debug_log.append(f"[Player 2] Confidence: {p2.get_confidence_percentage():.1f}%")

            # Log game result
            self.debug_log.append(f"\n[Game] Status: {game_result.status.upper()}")
            if game_result.status == 'valid':
                self.debug_log.append(f"[Game] Winner: {game_result.winner.upper()}")
            elif game_result.reason:
                self.debug_log.append(f"[Game] Reason: {game_result.reason}")

            # Format result for display
            class_text, conf_text = self._format_game_result(game_result)

            self.debug_log.append("\n[Game Mode] ✓ Complete!")

        except Exception as e:
            self.debug_log.append(f"\n[ERROR] {str(e)}")
            import traceback
            self.debug_log.append(traceback.format_exc())
            class_text, conf_text = self.result_step.set_error()

        # Display mode (game mode uses annotated by default)
        if display_mode == "Original":
            output_image = self.current_image
        elif display_mode == "Annotated":
            output_image = self.annotated_image if self.annotated_image is not None else self.current_image
        else:  # Preprocessed
            output_image = self.current_image  # Game mode doesn't have preprocessed for full image

        pipeline_info_text = self.pipeline_info.update(self.pipeline.get_pipeline_info())

        return (
            output_image,
            class_text,
            conf_text,
            pipeline_info_text,
            self.debug_log.get_text()
        )

    def _format_game_result(self, game_result):
        """Format game result for UI display."""
        p1_class = game_result.player1_result.predicted_class.value.upper()
        p2_class = game_result.player2_result.predicted_class.value.upper()
        p1_conf = game_result.player1_result.get_confidence_percentage()
        p2_conf = game_result.player2_result.get_confidence_percentage()

        if game_result.status == 'invalid':
            class_text = f"❌ Invalid Game\n\nPlayer 1: {p1_class}\nPlayer 2: {p2_class}"
            conf_text = f"Reason:\n{game_result.reason}"
        elif game_result.status == 'draw':
            class_text = f"🤝 Draw!\n\nPlayer 1: {p1_class}\nPlayer 2: {p2_class}"
            conf_text = f"P1 conf: {p1_conf:.1f}%\nP2 conf: {p2_conf:.1f}%"
        else:  # valid
            winner = "Player 1" if game_result.winner == 'player1' else "Player 2"
            class_text = f"🎉 {winner} Wins!\n\nPlayer 1: {p1_class}\nPlayer 2: {p2_class}"
            conf_text = f"P1 conf: {p1_conf:.1f}%\nP2 conf: {p2_conf:.1f}%"

        return class_text, conf_text

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Rock-Paper-Scissors",
        ) as demo:

            create_header()

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Settings"):
                            # Game mode selector
                            with gr.Row():
                                game_mode_dropdown = gr.Dropdown(
                                    choices=["Single Hand", "Two Player Game"],
                                    value="Single Hand",
                                    label="🎮 Mode",
                                    info="Single Hand: classify one gesture | Two Player Game: evaluate RPS match"
                                )

                            with gr.Row(equal_height=True):
                                preproc_dropdown, feature_dropdown, classifier_dropdown = create_pipeline_steps_block(
                                    self.preprocessors,
                                    self.feature_extractors,
                                    self.classifiers
                                )
                                
                                with gr.Row(equal_height=False):
                                    input_image = self.input_section.create()
                                    process_btn = self.controls_section.create()

                        with gr.Tab("Logs"):
                            pipeline_out = self.pipeline_info.create()
                            log_out = self.debug_log.create()

                with gr.Column(scale=7):
                    output_image, display_radio = self.preview_section.create()
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            class_out = gr.Textbox(
                                label="",
                                value=self.result_step._current_class,
                                container=False,
                                interactive=False,
                                elem_classes="class-output-inline"
                            )
                            self.result_step.class_component = class_out
                        with gr.Column(scale=1):
                            conf_out = gr.Textbox(
                                label="",
                                value=self.result_step._current_confidence,
                                container=False,
                                interactive=False,
                                elem_classes="confidence-output-inline"
                            )
                            self.result_step.confidence_component = conf_out

            inputs = [
                input_image,
                preproc_dropdown,
                feature_dropdown,
                classifier_dropdown,
                display_radio,
                game_mode_dropdown
            ]
            outputs = [output_image, class_out, conf_out, pipeline_out, log_out]

            input_image.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            process_btn.click(fn=self.process_image, inputs=inputs, outputs=outputs)
            preproc_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            feature_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            classifier_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            display_radio.change(fn=self.process_image, inputs=inputs, outputs=outputs)
            game_mode_dropdown.change(fn=self.process_image, inputs=inputs, outputs=outputs)

        return demo