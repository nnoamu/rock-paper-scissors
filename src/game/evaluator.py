from typing import Dict, Tuple

from core.classification_result import ClassificationResult
from .game_result import GameResult


class GameEvaluator:
    """Kő-papír-olló játék kiértékelése két játékos között."""

    RULES: Dict[Tuple[str, str], str] = {
        ('ROCK', 'SCISSORS'): 'player1',
        ('SCISSORS', 'PAPER'): 'player1',
        ('PAPER', 'ROCK'): 'player1',
        ('SCISSORS', 'ROCK'): 'player2',
        ('PAPER', 'SCISSORS'): 'player2',
        ('ROCK', 'PAPER'): 'player2',
    }

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def evaluate(
        self,
        player1_result: ClassificationResult,
        player2_result: ClassificationResult
    ) -> GameResult:
        p1_class = player1_result.predicted_class
        p2_class = player2_result.predicted_class
        p1_conf = player1_result.confidence
        p2_conf = player2_result.confidence

        if p1_conf < self.min_confidence:
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason=f'Low confidence for Player 1 ({p1_conf:.2f} < {self.min_confidence})'
            )

        if p2_conf < self.min_confidence:
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason=f'Low confidence for Player 2 ({p2_conf:.2f} < {self.min_confidence})'
            )

        if p1_class == 'UNKNOWN':
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason='Unknown gesture from Player 1'
            )

        if p2_class == 'UNKNOWN':
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason='Unknown gesture from Player 2'
            )

        if p1_class == p2_class:
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='draw',
                reason=f'Both players showed {p1_class.value}'
            )

        winner = self.RULES.get((p1_class, p2_class))

        if winner is None:
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason=f'Invalid game combination: {p1_class.value} vs {p2_class.value}'
            )

        return GameResult(
            player1_result=player1_result,
            player2_result=player2_result,
            winner=winner,
            status='valid',
            reason=None
        )
