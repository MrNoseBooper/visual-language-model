"""Record samples of your own gestures for training.

Run with:  python src/collect.py

Each sample is one row in data/landmarks.csv: the gesture name, which hand, when it was taken,
and the 63 normalised numbers from features.normalise(). No photos are saved.

Keys (click the video window first):
  SPACE   save one sample
  B       burst mode on/off (saves a sample every BURST_INTERVAL seconds)
  1-9     switch to that gesture from gestures.txt
  Q       quit
"""
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from camera import Camera
from features import feature_names, normalise
from hands import HandTracker, draw_hand

ROOT = Path(__file__).resolve().parent.parent
GESTURES_FILE = ROOT / "gestures.txt"
DATA_FILE = ROOT / "data" / "landmarks.csv"
BURST_INTERVAL = 0.2        # seconds between samples in burst mode
SAVED_FLASH_TIME = 0.15     # seconds the green "saved" border stays on screen
TARGET_SAMPLES = 100        # count turns green when a gesture has this many samples
WINDOW_NAME = "Collect gestures (press Q to quit)"

TIPS = """
Tips for good samples:
  - Vary the ANGLE: tilt and rotate your hand a little, palm more towards / away from the camera.
  - Vary the DISTANCE: close to the camera and further away.
  - Use BOTH hands (left hands are mirrored automatically, but it still helps).
  - Vary LIGHTING and BACKGROUND if you can (daylight, lamp, different rooms).
  - Aim for 50-150 samples per gesture. Burst mode (B) while slowly moving your hand is quickest.
  - Record a "none" class too: relaxed hands, scratching, waving, half-made gestures, typing pose.
    This teaches the model when you are NOT making a gesture.
  - Only ONE hand in view while saving.
"""


def load_gestures():
    if not GESTURES_FILE.exists():
        return []
    return [line.strip() for line in GESTURES_FILE.read_text().splitlines() if line.strip()]


def choose_gesture(gestures):
    """Ask in the terminal which gesture to record. A new name is added to gestures.txt."""
    print("Gestures in gestures.txt:")
    for number, name in enumerate(gestures, start=1):
        print(f"  {number}. {name}")
    while True:
        answer = input("Type a number, or a new gesture name: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(gestures):
            return gestures[int(answer) - 1]
        if answer and not answer.isdigit():
            name = answer.lower().replace(" ", "_")
            if name not in gestures:
                with open(GESTURES_FILE, "a") as f:
                    f.write(name + "\n")
                gestures.append(name)
                print(f"Added '{name}' to gestures.txt")
            return name
        print("Please type one of the numbers above, or a name.")


def count_samples():
    """How many samples each gesture already has in the CSV."""
    if not DATA_FILE.exists():
        return Counter()
    with open(DATA_FILE, newline="") as f:
        return Counter(row["label"] for row in csv.DictReader(f))


def save_sample(label, hand):
    """Append one row to data/landmarks.csv (creating it, with a header row, if needed)."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    is_new_file = not DATA_FILE.exists()
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["label", "handedness", "timestamp", *feature_names()])
        features = normalise(hand.world_landmarks, hand.handedness)
        writer.writerow([
            label,
            hand.handedness,
            datetime.now().isoformat(timespec="seconds"),
            *[f"{value:.5f}" for value in features],
        ])


def draw_status(frame, label, count, burst, message):
    """Info panel at the bottom of the window."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 0), -1)
    count_colour = (0, 255, 0) if count >= TARGET_SAMPLES else (0, 200, 255)
    cv2.putText(frame, f"Gesture: {label}", (10, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Samples: {count}", (width - 190, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_colour, 2)
    burst_text = "ON" if burst else "off"
    cv2.putText(frame, f"SPACE save | B burst: {burst_text} | 1-9 switch | Q quit",
                (10, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if message:
        cv2.putText(frame, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main():
    gestures = load_gestures()
    label = choose_gesture(gestures)
    counts = count_samples()
    print(TIPS)
    print(f"Recording '{label}'. Click the video window, then use the keys shown at the bottom.")

    camera = Camera()
    tracker = HandTracker()   # loads the model once, before the loop
    burst = False
    last_saved = 0.0
    try:
        for frame in camera.frames():
            hands = tracker.detect(frame)
            for hand in hands:
                draw_hand(frame, hand)

            # Only save when exactly one hand is visible, so we know which hand the label belongs to.
            message = ""
            if len(hands) == 0:
                message = "No hand in view"
            elif len(hands) > 1:
                message = "Show only ONE hand to save"

            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            want_save = key == ord(" ") or (burst and now - last_saved >= BURST_INTERVAL)

            if want_save and len(hands) == 1:
                save_sample(label, hands[0])
                counts[label] += 1
                last_saved = now

            if key in (ord("b"), ord("B")):
                burst = not burst
            elif ord("1") <= key <= ord("9"):
                number = key - ord("1")
                if number < len(gestures):
                    label = gestures[number]
                    burst = False   # stop bursting so the new gesture doesn't get wrong samples
                    print(f"Switched to '{label}'")
            elif key in (ord("q"), ord("Q")):
                break

            # Green border for a moment after each save, so you can see it worked.
            if now - last_saved < SAVED_FLASH_TIME:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 0), 8)

            draw_status(frame, label, counts[label], burst, message)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    print("\nSamples per gesture so far:")
    for name in gestures:
        print(f"  {name:15s} {counts[name]}")
    print(f"Saved in {DATA_FILE}")


if __name__ == "__main__":
    main()
