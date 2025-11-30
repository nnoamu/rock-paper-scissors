"""
Factory function for creating preprocessing modules from type names and parameters.
"""

from typing import Dict, Any, Optional
from .base_processor import PreprocessingModule
from preprocessing import (
    IdentityPreprocessor,
    DownscalePreprocessor,
    GrayscaleConverter,
    ChannelSplitter,
    GaussianBlurModule,
    EdgeDetectorModule,
    SkinColorSegmenterModule
)


def create_preprocessing_module(step_type: str, parameters: Dict[str, Any] = None) -> Optional[PreprocessingModule]:
    """
    Create a preprocessing module from type name and parameters.
    
    Args:
        step_type: Type name of the preprocessing module
        parameters: Dictionary of parameters for the module
    
    Returns:
        PreprocessingModule instance or None if type not found
    """
    if parameters is None:
        parameters = {}
    
    if step_type == "None":
        return IdentityPreprocessor()
    elif step_type == "Downscale (640px)":
        return DownscalePreprocessor(max_size=640)
    elif step_type == "Downscale (480px)":
        return DownscalePreprocessor(max_size=480)
    elif step_type == "Grayscale":
        return GrayscaleConverter()
    elif step_type == "Split":
        return ChannelSplitter()
    elif step_type == "Blur":
        return GaussianBlurModule()
    elif step_type == "Edge detection":
        return EdgeDetectorModule(
            lower_thresh=int(parameters.get("lower_thresh", 0)),
            upper_thresh=int(parameters.get("upper_thresh", 40))
        )
    elif step_type == "Downscale":
        return DownscalePreprocessor(
            max_size=parameters.get("max_size", 640)
        )
    elif step_type == "ChannelSplitter":
        return ChannelSplitter()
    elif step_type == "GaussianBlur":
        return GaussianBlurModule()
    elif step_type == "EdgeDetector":
        return EdgeDetectorModule(
            lower_thresh=int(parameters.get("lower_thresh", 0)),
            upper_thresh=int(parameters.get("upper_thresh", 40))
        )
    elif step_type == "SkinColorSegmenter":
        return SkinColorSegmenterModule(
            parameters.get("model_path", "models/skin_segmentation/model1")
        )
    else:
        print(f"Warning: Unknown preprocessing type: {step_type}")
        return None


def get_preprocessing_type_names() -> Dict[str, str]:
    """
    Get list of available preprocessing type names.
    
    Returns:
        Dictionary mapping internal names to display names (same in this case)
    """
    return {
        "None": "None",
        "Downscale (640px)": "Downscale (640px)",
        "Downscale (480px)": "Downscale (480px)",
        "Grayscale": "Grayscale",
        "Split": "Split",
        "Blur": "Blur",
        "Edge detection": "Edge detection",
    }
