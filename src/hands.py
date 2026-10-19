"""Hand tracking with MediaPipe's Hand Landmarker (Tasks API).

For each hand in a frame it finds 21 points ("landmarks"): the wrist, knuckles and fingertips.
Each point has x and y (0..1, as a fraction of the image width/height) and z (depth, relative
to the wrist; smaller = closer to the camera).

Model file:  models/hand_landmarker.task
Downloaded from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5   # how sure it must be to find a new hand
MIN_TRACKING_CONFIDENCE = 0.5    # how sure it must be to keep following a hand
# MediaPipe's docs say its Left/Right labels are right for a mirrored image, but testing on this
# laptop showed them reversed, so we swap them. Set to False if they ever come out wrong again.
SWAP_HANDEDNESS = True

# Drawing colours (OpenCV uses Blue, Green, Red order)
LEFT_COLOUR = (255, 128, 0)      # blue-ish for the left hand
RIGHT_COLOUR = (0, 128, 255)     # orange for the right hand
POINT_COLOUR = (255, 255, 255)   # white dots


@dataclass
class Hand:
    landmarks: np.ndarray   # shape (21, 3): x, y, z for each of the 21 points
    handedness: str         # "Left" or "Right"
    score: float            # how sure MediaPipe is about Left/Right (0..1)


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, max_hands=MAX_HANDS):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Download it from the URL at the top of hands.py."
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode tracks hands from frame to frame, which is smoother and faster
            # than detecting them from scratch in every frame.
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        # Load the model once here, never per frame.
        self.landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame):
        """Find hands in a frame. Returns a list of Hand (empty list if no hands).

        The frame should be mirrored (selfie view), as Camera.frames() gives it.
        Left/Right then refer to your real hands (see SWAP_HANDEDNESS).
        """
        # OpenCV gives BGR colour order; MediaPipe wants RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a timestamp that always goes up.
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(image, timestamp_ms)

        hands = []
        for points, handedness in zip(result.hand_landmarks, result.handedness):
            landmarks = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            best = handedness[0]   # the most likely label
            label = best.category_name
            if SWAP_HANDEDNESS:
                label = "Right" if label == "Left" else "Left"
            hands.append(Hand(landmarks, label, best.score))
        return hands

    def close(self):
        self.landmarker.close()


def draw_hand(frame, hand):
    """Draw the hand skeleton (lines + dots) and a Left/Right label on the frame."""
    height, width = frame.shape[:2]
    # Convert 0..1 coordinates into pixel positions.
    pixels = [(int(x * width), int(y * height)) for x, y, _ in hand.landmarks]
    colour = LEFT_COLOUR if hand.handedness == "Left" else RIGHT_COLOUR

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pixels[connection.start], pixels[connection.end], colour, 2)
    for point in pixels:
        cv2.circle(frame, point, 4, POINT_COLOUR, -1)

    # Put the label just above the wrist (point 0).
    wrist_x, wrist_y = pixels[0]
    cv2.putText(frame, hand.handedness, (wrist_x - 30, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
