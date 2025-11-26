"""
Kep utility fuggvenyek Dear PyGui-hoz.
"""

import numpy as np
import cv2


def numpy_to_dpg_texture(image: np.ndarray) -> np.ndarray:
    """
    Numpy kep (BGR/Grayscale) konvertalasa Dear PyGui texture formatumra.

    Dear PyGui RGBA float32 formatumot var (0.0-1.0 ertekekkel).
    """
    if image is None:
        return None

    # Grayscale -> BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # BGR -> RGBA
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # Normalize 0-1 float32
    image = image.astype(np.float32) / 255.0

    # Flatten for Dear PyGui
    return image.flatten()


def resize_with_aspect(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Kep atmeretezese aspektus arany megtartasaval.
    """
    if image is None:
        return None

    h, w = image.shape[:2]

    # Szamold ki a skalazasi faktort
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    # Ha mar belefer, ne meretezz
    if scale >= 1.0:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_placeholder_image(width: int, height: int, text: str = "No Image") -> np.ndarray:
    """
    Placeholder kep letrehozasa szoveggel.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (45, 45, 48)  # BG_MEDIUM szin

    # Szoveg kozepre
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_w) // 2
    y = (height + text_h) // 2

    cv2.putText(image, text, (x, y), font, font_scale, (128, 128, 128), thickness)

    return image
