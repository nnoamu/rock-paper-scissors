"""
Preprocessing Pipeline - Manages multiple preprocessing steps in sequence.

Data structure: List of {id, type, enabled, parameters}
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Callable
from .base_processor import PreprocessingModule
from preprocessing import IdentityPreprocessor


class PreprocessingStep:
    """Represents a single preprocessing step in the pipeline."""
    
    def __init__(self, step_id: str, step_type: str, enabled: bool = True, parameters: Dict[str, Any] = None):
        self.id = step_id
        self.type = step_type
        self.enabled = enabled
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'enabled': self.enabled,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingStep':
        """Create step from dictionary."""
        return cls(
            step_id=data.get('id', str(uuid.uuid4())),
            step_type=data['type'],
            enabled=data.get('enabled', True),
            parameters=data.get('parameters', {})
        )


class PreprocessingPipeline:
    """Manages a sequence of preprocessing steps."""
    
    def __init__(self, step_factory: Optional[Callable[[str, Dict[str, Any]], PreprocessingModule]] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            step_factory: Function that creates a PreprocessingModule from (type_name, parameters)
        """
        self.steps: List[PreprocessingStep] = []
        self.step_factory = step_factory
    
    def add_step(self, step_type: str, enabled: bool = True, parameters: Dict[str, Any] = None, position: Optional[int] = None) -> str:
        """
        Add a new preprocessing step.
        
        Args:
            step_type: Type name of the preprocessing module
            enabled: Whether the step is enabled (all enabled by def)
            position: Position to insert at (None = append to end)
        
        Returns:
            The ID of the newly created step
        """
        step_id = str(uuid.uuid4())
        step = PreprocessingStep(step_id, step_type, enabled, parameters)
        
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        return step_id
    
    def get_step(self, step_id: str) -> Optional[PreprocessingStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def clear(self):
        """Clear all steps."""
        self.steps.clear()
    
    def reset_to_default(self):
        """Reset pipeline to default (empty - no steps)."""
        self.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary for JSON serialization."""
        return {
            'steps': [step.to_dict() for step in self.steps]
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load pipeline from dictionary."""
        self.clear()
        for step_data in data.get('steps', []):
            self.steps.append(PreprocessingStep.from_dict(step_data))
    
    def to_json(self) -> str:
        """Serialize pipeline to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def from_json(self, json_str: str):
        """Load pipeline from JSON string."""
        data = json.loads(json_str)
        self.from_dict(data)
    
    def build_modules(self) -> List[PreprocessingModule]:
        if not self.step_factory:
            raise RuntimeError("Step factory not set. Cannot build modules.")
        
        modules = []
        for step in self.steps:
            if step.enabled:
                try:
                    module = self.step_factory(step.type, step.parameters)
                    if module is not None:
                        modules.append(module)
                except Exception as e:
                    # Log error but continue with other steps
                    print(f"Warning: Failed to create preprocessing module '{step.type}': {e}")
        
        # If no enabled steps, return empty list (no preprocessing)
        # The pipeline will use the original image
        
        return modules
    
    def get_pipeline_info(self) -> str:
        if not self.steps:
            return "Preprocessing: None"
        
        enabled_steps = [step for step in self.steps if step.enabled]
        if not enabled_steps:
            return "Preprocessing: None (all disabled)"
        
        step_names = [step.type for step in enabled_steps]
        return f"Preprocessing: {' → '.join(step_names)}"

