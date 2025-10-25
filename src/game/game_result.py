from dataclasses import dataclass
from typing import Optional

from core.classification_result import ClassificationResult


@dataclass
class GameResult:
    """Két játékos közötti kő-papír-olló játék eredménye."""
    player1_result: ClassificationResult
    player2_result: ClassificationResult
    winner: Optional[str]
    status: str
    reason: Optional[str] = None

    def __str__(self) -> str:
        if self.status == 'invalid':
            return f"Invalid Game - {self.reason}"
        elif self.status == 'draw':
            return f"Draw! Both players: {self.player1_result.predicted_class}"
        else:
            return f"{self.winner} wins! ({self.player1_result.predicted_class} vs {self.player2_result.predicted_class})"
