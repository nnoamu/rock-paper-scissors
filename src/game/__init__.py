"""
Game module - two-player rock-paper-scissors game evaluation.
"""

from .game_result import GameResult
from .evaluator import GameEvaluator
from .wrapper import TwoPlayerGameWrapper

__all__ = ['GameResult', 'GameEvaluator', 'TwoPlayerGameWrapper']
