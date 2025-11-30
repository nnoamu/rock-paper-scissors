from typing import Optional

from core.classification_result import ClassificationResult
from .game_result import GameResult


class GameEvaluator:

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def _determine_winner(self, player1_gesture: str, player2_gesture: str) -> Optional[str]:
        p1 = player1_gesture.upper()
        p2 = player2_gesture.upper()

        if p1 == p2:
            return None

        if p1 == 'ROCK' and p2 == 'SCISSORS':
            return 'player1'
        if p1 == 'SCISSORS' and p2 == 'PAPER':
            return 'player1'
        if p1 == 'PAPER' and p2 == 'ROCK':
            return 'player1'

        return 'player2'

    def _get_gesture_string(self, classification_result: ClassificationResult) -> str:
        gesture = classification_result.predicted_class

        if hasattr(gesture, 'value'):
            return str(gesture.value).upper()
        elif hasattr(gesture, 'name'):
            return str(gesture.name).upper()
        else:
            return str(gesture).upper()

    def evaluate(
            self,
            player1_result: ClassificationResult,
            player2_result: ClassificationResult
    ) -> GameResult:

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

        p1_gesture = self._get_gesture_string(player1_result)
        p2_gesture = self._get_gesture_string(player2_result)

        if p1_gesture == 'UNKNOWN':
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason='Unknown gesture from Player 1'
            )

        if p2_gesture == 'UNKNOWN':
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='invalid',
                reason='Unknown gesture from Player 2'
            )

        print(f"DEBUG: Player 1 = {p1_gesture}, Player 2 = {p2_gesture}")

        winner = self._determine_winner(p1_gesture, p2_gesture)

        if winner is None:
            return GameResult(
                player1_result=player1_result,
                player2_result=player2_result,
                winner=None,
                status='draw',
                reason=f'Both players showed {p1_gesture}'
            )

        return GameResult(
            player1_result=player1_result,
            player2_result=player2_result,
            winner=winner,
            status='valid',
            reason=None
        )
