"""
Desktop UI komponensek.
"""

from desktop_ui.components.camera import CameraManager
from desktop_ui.components.header import HeaderPanel
from desktop_ui.components.pipeline_panel import PipelinePanel
from desktop_ui.components.result_panel import ResultPanel
from desktop_ui.components.preview import PreviewPanel
from desktop_ui.components.debug_log import DebugLogPanel, PipelineInfoPanel

__all__ = [
    'CameraManager',
    'HeaderPanel',
    'PipelinePanel',
    'ResultPanel',
    'PreviewPanel',
    'DebugLogPanel',
    'PipelineInfoPanel'
]
