"""
Preprocessing Pipeline UI Component - Manages multiple preprocessing steps.
"""

import gradio as gr
import json
from typing import List, Dict, Any, Optional, Callable
from core.preprocessing_pipeline import PreprocessingPipeline, PreprocessingStep
from ui.utils import render

def create_preprocessing_pipeline_ui(preprocessor_types: Dict[str, Any], 
                                     step_factory: Callable[[str, Dict[str, Any]], Any]):
    """
    Create preprocessing pipeline UI with multiple steps.
    """
    from core.preprocessing_factory import get_preprocessing_type_names
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(step_factory=step_factory)
    pipeline.reset_to_default()
    
    # Get available type names
    type_names = get_preprocessing_type_names()
    available_types = list(type_names.keys())
    if not available_types:
        available_types = list(preprocessor_types.keys())
    
    with gr.Column(scale=1):
        with gr.Group(elem_classes="pipeline-step-small active"):
            gr.HTML(render(
                'step_header.html',
                number=1,
                title="Preprocessing",
                disabled=False
            ))
            with gr.Group(elem_classes="step-content"):
                pipeline_state = gr.State(value=pipeline.to_json())
                
                # Hidden state to track which step and action
                action_state = gr.State(value={"step_id": "", "action": ""})
                
                steps_container = gr.HTML(
                    value=render_pipeline_steps_html(pipeline.steps),
                    elem_classes="pipeline-steps-display"
                )
                
                with gr.Row():
                    add_step_dropdown = gr.Dropdown(
                        choices=available_types,
                        value=available_types[0] if available_types else None,
                        label="",
                        container=False,
                        scale=3,
                        min_width=150
                    )
                    add_step_btn = gr.Button("+ Add Step", scale=1, size="sm", min_width=100)
                
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset", scale=1, size="sm", min_width=80)
    
    return (pipeline_state, steps_container, add_step_dropdown, add_step_btn, 
            reset_btn)


def add_step_handler(step_type: str, current_state: str, preprocessor_types: Dict[str, Any],
                     step_factory: Callable) -> tuple:
    """Add a new step to the pipeline."""
    pipeline = PreprocessingPipeline(step_factory=step_factory)
    if current_state:
        pipeline.from_json(current_state)
    pipeline.add_step(step_type, enabled=True, parameters={})
    new_state = pipeline.to_json()
    
    html = render_pipeline_steps_html(pipeline.steps)
    return new_state, html


def reset_handler(current_state: str, step_factory: Callable) -> tuple:
    """Reset pipeline to default."""
    pipeline = PreprocessingPipeline(step_factory=step_factory)
    pipeline.reset_to_default()
    new_state = pipeline.to_json()
    html = render_pipeline_steps_html(pipeline.steps)
    return new_state, html


def render_pipeline_steps_html(steps: List[PreprocessingStep]) -> str:
    """Render HTML display of pipeline steps."""
    if not steps:
        return "<div class='no-steps'>No steps. Click 'Add Step' to add one.</div>"
    
    html = "<div class='pipeline-steps-list'>"
    for i, step in enumerate(steps):
        params_str = " "
        if step.parameters:
            params_str = ", ".join([f"{k}={v}" for k, v in step.parameters.items()])
        
        html += f"""
        <div class="preprocessing-step-display" data-step-id="{step.id}">
            <span class="step-index">{i + 1}.</span>
            <span class="step-type"><strong>{step.type}</strong></span>
        </div>
        """
    html += "</div>"
    return html
