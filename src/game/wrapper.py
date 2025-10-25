import cv2
import numpy as np

from core.pipeline import ProcessingPipeline
from .evaluator import GameEvaluator
from .game_result import GameResult


class TwoPlayerGameWrapper:
    """Pipeline wrapper két játékos mód támogatásához. Képet kettévágja, mindkét félen lefuttatja a pipeline-t, majd kiértékeli a játékot."""

    def __init__(self, pipeline: ProcessingPipeline, min_confidence: float = 0.7):
        self.pipeline = pipeline
        self.evaluator = GameEvaluator(min_confidence=min_confidence)

    def run(self, image: np.ndarray) -> GameResult:
        game_result, _ = self.run_with_visualization(image)
        return game_result

    def run_with_visualization(self, image: np.ndarray) -> tuple:
        height, width = image.shape[:2]
        mid_point = width // 2

        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]

        _, _, player1_result, annotated_left = self.pipeline.process_full_pipeline(left_half)
        _, _, player2_result, annotated_right = self.pipeline.process_full_pipeline(right_half)

        game_result = self.evaluator.evaluate(player1_result, player2_result)

        annotated_combined = self._combine_annotated_halves(
            annotated_left, annotated_right, game_result
        )

        return game_result, annotated_combined

    def _combine_annotated_halves(self, left_img: np.ndarray, right_img: np.ndarray,
                                   game_result: GameResult) -> np.ndarray:
        if len(left_img.shape) == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
        if len(right_img.shape) == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)

        combined = np.hstack([left_img, right_img])

        h, w = combined.shape[:2]
        mid_point = w // 2

        cv2.line(combined, (mid_point, 0), (mid_point, h), (255, 255, 255), 3)

        p1_color, p2_color = self._get_player_border_colors(game_result)

        base_size = 480.0
        scale = min(w, h) / base_size
        scale = max(0.5, min(scale, 3.0))

        border_thickness = max(3, int(8 * scale))
        text_scale = max(0.5, min(0.7 * scale, 2.0))
        text_thickness = max(1, int(1.5 * scale))
        text_padding = max(10, int(15 * scale))
        text_offset_bottom = max(20, int(30 * scale))

        cv2.rectangle(combined, (0, 0), (mid_point - 2, h - 1), p1_color, border_thickness)
        cv2.rectangle(combined, (mid_point + 2, 0), (w - 1, h - 1), p2_color, border_thickness)
        cv2.putText(combined, "Player 1", (text_padding, h - text_offset_bottom),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
        cv2.putText(combined, "Player 2", (mid_point + text_padding, h - text_offset_bottom),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

        return combined

    def _get_player_border_colors(self, game_result: GameResult) -> tuple:
        if game_result.status == 'invalid':
            return (0, 0, 255), (0, 0, 255)
        elif game_result.status == 'draw':
            return (0, 255, 255), (0, 255, 255)
        else:
            if game_result.winner == 'player1':
                return (0, 255, 0), (0, 0, 255)
            else:
                return (0, 0, 255), (0, 255, 0)
