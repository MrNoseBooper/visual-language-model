"""Turns a hand's 21 points into 63 numbers that describe its SHAPE only.

collect.py, train.py and app.py all use normalise() from here, so the model always sees
numbers prepared in exactly the same way.
"""
import numpy as np

NUM_POINTS = 21
NUM_FEATURES = NUM_POINTS * 3   # x, y, z for each point
WRIST = 0                       # MediaPipe's point 0 is the wrist


def normalise(world_landmarks, handedness):
    """Return 63 numbers for one hand, independent of where the hand is, how big it is, and which hand it is.

    world_landmarks: (21, 3) array from Hand.world_landmarks
    handedness:      "Left" or "Right"
    """
    points = np.asarray(world_landmarks, dtype=np.float32).reshape(NUM_POINTS, 3).copy()

    # 1. Position: measure every point from the wrist, so the wrist is always (0, 0, 0).
    points -= points[WRIST]

    # 2. Which hand: mirror left hands left-to-right so they look like right hands.
    #    Then one model works for both hands.
    if handedness == "Left":
        points[:, 0] *= -1

    # 3. Size: scale so the point furthest from the wrist is exactly 1 away.
    #    Big and small hands then give the same numbers.
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points /= scale

    return points.flatten()   # [x0, y0, z0, x1, y1, z1, ...]


def feature_names():
    """Column names for the 63 numbers: x0, y0, z0, x1, ..."""
    return [f"{axis}{i}" for i in range(NUM_POINTS) for axis in "xyz"]
