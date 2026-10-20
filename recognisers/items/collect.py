"""Record photos of your items for training.

Run with:  python main.py items collect

Saves the part of the picture the recogniser looks at (around your hand, or the centre; press C)
to data/<label>/<time>.jpg in this folder. Doesn't need CLIP or PyTorch, so it starts quickly.

Keys (click the video window first):
  SPACE   save one photo
  B       burst mode on/off (saves a photo every BURST_INTERVAL seconds)
  1-9     switch to that label from labels.txt
  C       crop mode: Auto (hand, else centre) -> Hand only -> Centre only
  Q       quit
"""
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

from common.camera import Camera
from common.hands import HandTracker
from common.labels import choose_label, load_labels

from . import FOLDER
from .crop import MODE_NAMES, choose_box, cut, draw_box, next_mode
from .predictor import LABELS_FILE

DATA_DIR = FOLDER / "data"
BURST_INTERVAL = 0.3        # seconds between photos in burst mode
SAVE_MAX_SIDE = 320         # photos are shrunk to at most this many pixels (CLIP only uses 224)
SAVED_FLASH_TIME = 0.15     # seconds the green "saved" border stays on screen
TARGET_PHOTOS = 100         # count turns green when a label has this many photos
IMAGE_TYPES = ("*.jpg", "*.jpeg", "*.png")
WINDOW_NAME = "Collect item photos (press Q to quit)"

TIPS = """
Tips for good photos:
  - Hold the item so it fills a good part of the green box (or put it in the centre box).
  - Vary the ANGLE: turn the item, show different sides, tilt it.
  - Vary the DISTANCE, LIGHTING and BACKGROUND (desk, wall, window, different rooms).
  - Use both hands and hold it in different ways.
  - Aim for 50-150 photos per item. Burst mode (B) while slowly turning the item is quickest.
  - Record "none" too: empty hands, random things that are NOT your items, just the background.
    This teaches the model when there's nothing to recognise.
"""


def count_photos(label):
    folder = DATA_DIR / label
    return sum(len(list(folder.glob(pattern))) for pattern in IMAGE_TYPES) if folder.exists() else 0


def save_photo(label, crop):
    """Shrink the crop if needed and save it as data/<label>/<time>.jpg."""
    scale = SAVE_MAX_SIDE / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    folder = DATA_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    # Time + a short random code, so two photos can never get the same name.
    path = folder / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}.jpg"
    # imencode + tofile instead of imwrite, so folder names with any characters work on Windows.
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        encoded.tofile(str(path))
    return ok


def draw_status(frame, label, count, burst, crop_mode, message):
    """Info panel at the bottom of the window."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 0), -1)
    count_colour = (0, 255, 0) if count >= TARGET_PHOTOS else (0, 200, 255)
    cv2.putText(frame, f"Item: {label}", (10, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Photos: {count}", (width - 180, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_colour, 2)
    burst_text = "ON" if burst else "off"
    cv2.putText(frame, f"SPACE save | B burst: {burst_text} | 1-9 switch | C crop | Q quit",
                (10, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Crop: {MODE_NAMES[crop_mode]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if message:
        cv2.putText(frame, message, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    labels = load_labels(LABELS_FILE)
    label = choose_label(LABELS_FILE, labels)
    counts = {name: count_photos(name) for name in labels}
    print(TIPS)
    print(f"Recording '{label}'. Click the video window, then use the keys shown at the bottom.")

    camera = Camera()
    tracker = HandTracker()   # loads the hand model once, before the loop
    crop_mode = "auto"
    burst = False
    last_saved = 0.0
    try:
        for frame in camera.frames():
            hands = tracker.detect(frame) if crop_mode != "centre" else []
            box, source = choose_box(crop_mode, hands, frame)
            crop = cut(frame, box) if box is not None else None   # cut BEFORE drawing on the frame
            message = "" if box is not None else "Show your hand to save (Hand only mode)"

            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            want_save = key == ord(" ") or (burst and now - last_saved >= BURST_INTERVAL)
            if want_save and crop is not None and save_photo(label, crop):
                counts[label] = counts.get(label, 0) + 1
                last_saved = now

            if key in (ord("b"), ord("B")):
                burst = not burst
            elif ord("1") <= key <= ord("9"):
                number = key - ord("1")
                if number < len(labels):
                    label = labels[number]
                    burst = False   # stop bursting so the new label doesn't get wrong photos
                    print(f"Switched to '{label}'")
            elif key in (ord("c"), ord("C")):
                crop_mode = next_mode(crop_mode)
            elif key in (ord("q"), ord("Q")):
                break

            if box is not None:
                draw_box(frame, box, source)
            # Green border for a moment after each save, so you can see it worked.
            if now - last_saved < SAVED_FLASH_TIME:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 0), 8)
            draw_status(frame, label, counts.get(label, 0), burst, crop_mode, message)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nPhotos so far ({DATA_DIR}):")
    for name in labels:
        print(f"  {name:15s} {counts.get(name, 0)}")
