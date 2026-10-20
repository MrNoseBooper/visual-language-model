"""Picks which part of the camera picture the item recogniser looks at.

Looking at a smaller area around the item stops the model being distracted by the background.
Three modes (press C to switch):
  auto    around your hand(s) if visible, otherwise the centre of the view
  hand    only around your hand(s); nothing is recognised without a hand
  centre  always the centre of the view
"""
import cv2
import numpy as np

CROP_MODES = ("auto", "hand", "centre")
MODE_NAMES = {"auto": "Auto (hand, else centre)", "hand": "Hand only", "centre": "Centre only"}
HAND_EXPAND = 1.6        # the box around the hand(s) is made this much bigger, so the held item fits
CENTRE_SIZE = 0.7        # side of the centre box, as a share of the frame height
MIN_BOX = 96             # pixels: never look at an area smaller than this
HAND_BOX_COLOUR = (0, 255, 0)      # green box around the hand
CENTRE_BOX_COLOUR = (255, 255, 0)  # light blue box in the centre


def next_mode(mode):
    return CROP_MODES[(CROP_MODES.index(mode) + 1) % len(CROP_MODES)]


def _square(cx, cy, side, width, height):
    """A square box centred on (cx, cy), cut off at the frame edges. Returns (x1, y1, x2, y2)."""
    half = max(side, MIN_BOX) / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(width, cx + half)), int(min(height, cy + half))
    return x1, y1, x2, y2


def hand_box(hands, width, height):
    """Square box around all visible hands, enlarged by HAND_EXPAND."""
    xs = np.concatenate([hand.landmarks[:, 0] for hand in hands]) * width
    ys = np.concatenate([hand.landmarks[:, 1] for hand in hands]) * height
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * HAND_EXPAND
    return _square((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, side, width, height)


def centre_box(width, height):
    return _square(width / 2, height / 2, height * CENTRE_SIZE, width, height)


def choose_box(mode, hands, frame):
    """Returns (box, "hand" or "centre"), or (None, None) in "hand" mode when no hand is visible."""
    height, width = frame.shape[:2]
    if hands and mode in ("auto", "hand"):
        return hand_box(hands, width, height), "hand"
    if mode == "hand":
        return None, None
    return centre_box(width, height), "centre"


def cut(frame, box):
    """Copy of the part of the frame inside the box. Take it BEFORE drawing anything on the frame."""
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def draw_box(frame, box, source, text=None):
    """Draw the box, with optional text on a dark strip above it."""
    x1, y1, x2, y2 = box
    colour = HAND_BOX_COLOUR if source == "hand" else CENTRE_BOX_COLOUR
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    if text:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        top = max(y1, text_h + 10)
        cv2.rectangle(frame, (x1, top - text_h - 8), (x1 + text_w + 8, top), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 4, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
