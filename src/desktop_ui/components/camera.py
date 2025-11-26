"""
Kamera kezeles OpenCV-vel, real-time stream.
"""

import cv2
import numpy as np
import threading
from typing import Optional, Callable
import time


class CameraManager:
    """
    Real-time kamera kezelo osztaly.
    Kulon szalban fut a kamera olvasas a folyamatos FPS-ert.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        self.on_frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self.fps = 0
        self._frame_count = 0
        self._fps_start_time = time.time()

    def start(self) -> bool:
        """Kamera inditasa."""
        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[Camera] Nem sikerult megnyitni a kamerat (index: {self.camera_index})")
            return False

        # Optimalis beallitasok
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimalis buffer a latencsiert

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("[Camera] Kamera elindult")
        return True

    def stop(self):
        """Kamera leallitasa."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        with self.frame_lock:
            self.current_frame = None

        print("[Camera] Kamera leallitva")

    def _capture_loop(self):
        """Folyamatos kamera olvasas kulon szalban."""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Horizontalis tukorzes (termeszetesebb)
                frame = cv2.flip(frame, 1)

                with self.frame_lock:
                    self.current_frame = frame

                # FPS szamolas
                self._frame_count += 1
                elapsed = time.time() - self._fps_start_time
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_start_time = time.time()

                # Callback ha van
                if self.on_frame_callback:
                    self.on_frame_callback(frame)
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Aktualis frame lekerese (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Pillanatkep keszitese (masolat)."""
        return self.get_frame()

    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """Frame callback beallitasa."""
        self.on_frame_callback = callback

    def is_active(self) -> bool:
        """Kamera aktiv-e?"""
        return self.is_running and self.cap is not None and self.cap.isOpened()

    def get_fps(self) -> float:
        """Aktualis FPS lekerese."""
        return self.fps

    def __del__(self):
        """Destruktor - biztonsagos leallitas."""
        self.stop()
