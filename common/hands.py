"""Hand tracking with MediaPipe's Hand Landmarker (Tasks API).

For each hand in a frame it finds:
- 21 points ("landmarks"): the wrist, knuckles and fingertips. Each point has x and y (0..1, as a
  fraction of the image width/height) and z (depth, relative to the wrist; smaller = closer).
- Left or Right hand.
It does NOT name gestures: that's done by the recognisers (recognisers/gestures, and recognisers/items uses
it to crop around your hand).

Model file:  common/models/hand_landmarker.task
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

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5   # how sure it must be to find a new hand
MIN_TRACKING_CONFIDENCE = 0.5    # how sure it must be to keep following a hand
# MediaPipe's docs say its Left/Right labels are right for a mirrored image, but testing on this
# laptop showed them reversed, so we swap them. Set to False if they ever come out wrong again.
SWAP_HANDEDNESS = True

# Smoothing (see LabelSmoother below)
SCORE_THRESHOLD = 0.6            # below this score, show "Not sure"
STABLE_FRAMES = 5                # a gesture must win this many frames in a row before it's shown

# Drawing colours (OpenCV uses Blue, Green, Red order)
LEFT_COLOUR = (255, 128, 0)      # blue-ish for the left hand
RIGHT_COLOUR = (0, 128, 255)     # orange for the right hand
POINT_COLOUR = (255, 255, 255)   # white dots


@dataclass
class Hand:
    landmarks: np.ndarray   # shape (21, 3): x, y, z for each of the 21 points (0..1 image coordinates, for drawing)
    world_landmarks: np.ndarray  # shape (21, 3): the same points in metres around the hand's centre (for features)
    handedness: str         # "Left" or "Right" (your real hand)
    score: float            # how sure MediaPipe is about Left/Right (0..1)
    aspect: float           # frame width / height, to turn 0..1 image coordinates into equal units


def pair_hands(hands):
    """For two-hand gestures: return (left_hand, right_hand), or None unless exactly 2 hands are visible.

    Uses the Left/Right labels. If MediaPipe labels both hands the same (it sometimes does), it falls
    back to screen position: in the mirrored view your right hand is on the right of the screen.
    """
    if len(hands) != 2:
        return None
    a, b = hands
    if a.handedness != b.handedness:
        return (a, b) if a.handedness == "Left" else (b, a)
    return (a, b) if a.landmarks[0, 0] < b.landmarks[0, 0] else (b, a)


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
        """Find hands in a frame. Returns a list of Hand (empty if no hands).

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
        aspect = frame.shape[1] / frame.shape[0]

        hands = []
        for points, world_points, handedness in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness
        ):
            landmarks = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            world_landmarks = np.array([[p.x, p.y, p.z] for p in world_points], dtype=np.float32)
            side = handedness[0]   # the most likely label
            label = side.category_name
            if SWAP_HANDEDNESS:
                label = "Right" if label == "Left" else "Left"
            hands.append(Hand(landmarks, world_landmarks, label, side.score, aspect))
        return hands

    def close(self):
        self.landmarker.close()


class LabelSmoother:
    """Stops the displayed label (gesture or item) from flickering.

    Keeps a separate memory for each hand ("Left" / "Right" / "Both"). A new gesture is only shown
    after it has won STABLE_FRAMES frames in a row. Low-score guesses count as "Not sure".
    """

    def __init__(self, threshold=SCORE_THRESHOLD, stable_frames=STABLE_FRAMES):
        self.threshold = threshold
        self.stable_frames = stable_frames
        self._state = {}   # hand side -> {"shown": ..., "candidate": ..., "count": ...}

    def update(self, side, gesture, score):
        """Feed in this frame's guess for one hand; returns the gesture to display."""
        if score < self.threshold:
            gesture = "Not sure"

        state = self._state.setdefault(side, {"shown": "...", "candidate": None, "count": 0})
        if gesture == state["candidate"]:
            state["count"] += 1
        else:
            state["candidate"] = gesture
            state["count"] = 1

        if state["count"] >= self.stable_frames:
            state["shown"] = state["candidate"]
        return state["shown"]

    def forget_missing(self, present_sides):
        """Clear the memory of hands that left the view, so they start fresh when they come back."""
        for side in list(self._state):
            if side not in present_sides:
                del self._state[side]


def draw_hand(frame, hand, text=None):
    """Draw the hand skeleton (lines + dots), a Left/Right label, and optional text above the hand."""
    height, width = frame.shape[:2]
    # Convert 0..1 coordinates into pixel positions.
    pixels = [(int(x * width), int(y * height)) for x, y, _ in hand.landmarks]
    colour = LEFT_COLOUR if hand.handedness == "Left" else RIGHT_COLOUR

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pixels[connection.start], pixels[connection.end], colour, 2)
    for point in pixels:
        cv2.circle(frame, point, 4, POINT_COLOUR, -1)

    # Put the Left/Right label just below the wrist (point 0).
    wrist_x, wrist_y = pixels[0]
    cv2.putText(frame, hand.handedness, (wrist_x - 30, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    if text:
        draw_label_above(frame, [hand], text, colour)


def draw_label_above(frame, hands, text, colour):
    """Text on a dark box, centred just above the highest point of one or more hands."""
    height, width = frame.shape[:2]
    xs = [x * width for hand in hands for x in hand.landmarks[:, 0]]
    ys = [y * height for hand in hands for y in hand.landmarks[:, 1]]

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Centre over the hand(s), but keep the whole label inside the window.
    x = int((min(xs) + max(xs)) / 2 - text_w / 2)
    x = min(max(x, 4), width - text_w - 4)
    y = int(max(min(ys) - 15, text_h + 10))

    cv2.rectangle(frame, (x - 4, y - text_h - 6), (x + text_w + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness)
