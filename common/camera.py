"""A small reusable webcam helper (shared by every recogniser).

Opens the webcam, gives you mirrored frames one at a time, and keeps track of FPS
(frames per second) so you can see how fast the app is running.
"""
import time

import cv2

CAMERA_INDEX = 0        # 0 = the default (built-in) webcam
FRAME_WIDTH = 640       # requested size; the camera may pick the closest it supports
FRAME_HEIGHT = 480
FPS_SMOOTHING = 0.9     # 0..1, higher = steadier FPS number but slower to react


class Camera:
    def __init__(self, index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        # CAP_DSHOW uses Windows' DirectShow backend, which usually opens faster on Windows.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the webcam. Close other apps using it (Teams, Zoom, Camera) "
                "and check Windows Settings > Privacy & security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.fps = 0.0
        self._last_time = time.perf_counter()

    def frames(self):
        """Yield mirrored frames forever, until the camera stops giving them."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("The webcam stopped sending pictures (unplugged, or taken by another app?).")
                break

            # Flip left-right so it behaves like a mirror (selfie view).
            frame = cv2.flip(frame, 1)

            self._update_fps()
            yield frame

    def _update_fps(self):
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now
        # Blend the new value with the old one so the number doesn't jump around.
        if self.fps == 0.0:
            self.fps = instant_fps
        else:
            self.fps = FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * instant_fps

    def draw_fps(self, frame):
        """Write the current FPS in the top-left corner of the frame."""
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def release(self):
        """Close the webcam so other apps can use it."""
        self.cap.release()
