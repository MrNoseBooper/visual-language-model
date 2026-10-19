"""Record samples of your own gestures for training. One-hand and two-hand gestures are handled automatically.

Run with:  python src/collect.py

All gestures are listed in gestures.txt and saved to data/landmarks.csv. Each row is:
  gesture name, number of hands (1 or 2), which hand ("Left"/"Right", or "both"), time, and the numbers:
  - 1 hand visible  -> 63 numbers from features.normalise() (the remaining columns are left empty)
  - 2 hands visible -> 210 numbers from features.normalise_two()
train.py then works out which gestures are two-handed from how you recorded them. No photos are saved.

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
from features import NUM_FEATURES, NUM_TWO_HAND_FEATURES, normalise, normalise_two
from hands import HandTracker, draw_hand, pair_hands

ROOT = Path(__file__).resolve().parent.parent
GESTURES_FILE = ROOT / "gestures.txt"
DATA_FILE = ROOT / "data" / "landmarks.csv"
HEADER = ["label", "hands", "handedness", "timestamp", *[f"f{i}" for i in range(NUM_TWO_HAND_FEATURES)]]
BURST_INTERVAL = 0.2        # seconds between samples in burst mode
SAVED_FLASH_TIME = 0.15     # seconds the green "saved" border stays on screen
TARGET_SAMPLES = 100        # count turns green when a gesture has this many samples
PAIR_LINE_COLOUR = (0, 255, 255)   # yellow line between the wrists when both hands are found
WINDOW_NAME = "Collect gestures (press Q to quit)"

TIPS = """
Tips for good samples:
  - ONE-HAND gesture: keep your other hand out of view for most samples. Also record a few with your
    other hand visible but relaxed: that teaches the system it's still a one-hand gesture.
  - TWO-HAND gesture: keep both hands in view. Gestures recorded mostly with two hands become two-hand gestures.
  - Vary the ANGLE: tilt and rotate your hands a little, palms more towards / away from the camera.
  - Vary the DISTANCE: close to the camera and further away.
  - One-hand gestures: use BOTH hands (left hands are mirrored automatically, but it still helps).
  - Vary LIGHTING and BACKGROUND if you can (daylight, lamp, different rooms).
  - Aim for 50-150 samples per gesture. Burst mode (B) while slowly moving is quickest.
  - Record "none" with ONE hand and with TWO hands: relaxed hands, scratching, typing pose,
    half-made gestures. This teaches the model when you are NOT making a gesture.
"""


def load_gestures():
    if not GESTURES_FILE.exists():
        return []
    return [line.strip() for line in GESTURES_FILE.read_text().splitlines() if line.strip()]


def add_gesture(name, gestures):
    """Add a new gesture name to gestures.txt (and the list in memory)."""
    text = GESTURES_FILE.read_text() if GESTURES_FILE.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    GESTURES_FILE.write_text(text + name + "\n")
    gestures.append(name)
    print(f"Added '{name}' to gestures.txt")


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
                add_gesture(name, gestures)
            return name
        print("Please type one of the numbers above, or a name.")


def count_samples():
    """How many samples each gesture already has, counted as (gesture, number of hands)."""
    if not DATA_FILE.exists():
        return Counter()
    with open(DATA_FILE, newline="") as f:
        return Counter((row["label"], int(row["hands"])) for row in csv.DictReader(f))


def make_row(label, hands):
    """Build the CSV row for this frame from however many hands are visible.

    Returns (row, number of hands), or (None, 0) if no hand is visible.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    pair = pair_hands(hands)
    if pair is not None:
        left, right = pair
        features = normalise_two(left.world_landmarks, right.world_landmarks,
                                 left.landmarks, right.landmarks, left.aspect)
        return [label, 2, "both", timestamp, *[f"{value:.5f}" for value in features]], 2
    if len(hands) == 1:
        hand = hands[0]
        features = normalise(hand.world_landmarks, hand.handedness)
        empty = [""] * (NUM_TWO_HAND_FEATURES - NUM_FEATURES)
        return [label, 1, hand.handedness, timestamp, *[f"{value:.5f}" for value in features], *empty], 1
    return None, 0


def save_row(row):
    """Append one row to data/landmarks.csv (creating it, with a header row, if needed)."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    is_new_file = not DATA_FILE.exists()
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(HEADER)
        writer.writerow(row)


def draw_pair_line(frame, hands):
    """A yellow line between the wrists shows both hands were found (a two-hand sample)."""
    pair = pair_hands(hands)
    if pair is None:
        return
    height, width = frame.shape[:2]
    left_wrist, right_wrist = (pair[0].landmarks[0], pair[1].landmarks[0])
    cv2.line(frame, (int(left_wrist[0] * width), int(left_wrist[1] * height)),
             (int(right_wrist[0] * width), int(right_wrist[1] * height)), PAIR_LINE_COLOUR, 2)


def draw_status(frame, label, counts, burst, num_hands):
    """Info panel at the bottom of the window."""
    height, width = frame.shape[:2]
    one, two = counts[(label, 1)], counts[(label, 2)]
    cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 0), -1)
    count_colour = (0, 255, 0) if one + two >= TARGET_SAMPLES else (0, 200, 255)
    cv2.putText(frame, f"Gesture: {label}", (10, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"1-hand {one} | 2-hand {two}", (width - 260, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_colour, 2)
    burst_text = "ON" if burst else "off"
    cv2.putText(frame, f"SPACE save | B burst: {burst_text} | 1-9 switch | Q quit",
                (10, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    # Top-left: what a save would record right now.
    if num_hands == 0:
        cv2.putText(frame, "No hand in view", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Saving as {num_hands}-hand sample", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PAIR_LINE_COLOUR if num_hands == 2 else (255, 255, 255), 2)


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
            draw_pair_line(frame, hands)

            row, num_hands = make_row(label, hands)

            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            want_save = key == ord(" ") or (burst and now - last_saved >= BURST_INTERVAL)

            if want_save and row is not None:
                save_row(row)
                counts[(label, num_hands)] += 1
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

            draw_status(frame, label, counts, burst, num_hands)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nSamples so far ({DATA_FILE}):")
    print(f"  {'gesture':15s} {'1-hand':>7s} {'2-hand':>7s}")
    for name in gestures:
        print(f"  {name:15s} {counts[(name, 1)]:7d} {counts[(name, 2)]:7d}")


if __name__ == "__main__":
    main()
