"""Live item recognition, using only what was learned from YOUR photos.

Run with:  python main.py items run

Keys (click the video window first):
  C   crop mode: Auto (hand, else centre) -> Hand only -> Centre only
  R   reload the trained model (after training again)
  Q   quit

Nothing is recognised until you've collected photos and trained (python main.py items collect / train).
CLIP takes roughly 50-150 ms per picture on this laptop's CPU, so it runs in a background thread:
the video stays smooth and the answer updates as fast as CLIP can go.
"""
import threading
import time
from collections import deque

import cv2

from common.camera import Camera
from common.hands import HandTracker

from .crop import MODE_NAMES, choose_box, cut, draw_box, next_mode
from .predictor import NO_ITEM, ItemModel

WINDOW_NAME = "Item Recognizer (press Q to quit)"
AVERAGE_OVER = 5          # average the last few answers so the label doesn't flicker
NOT_SURE_BELOW = 0.4      # if the best answer is below this, show "Not sure"
TOP_N = 3                 # how many guesses to list on screen
TEXT_COLOUR = (255, 255, 255)
WARNING_COLOUR = (0, 0, 255)
NOT_TRAINED = "Not trained yet: collect photos, then train (python main.py items collect / train)"


class Worker(threading.Thread):
    """Runs CLIP + your model on the newest crop in the background. Old crops are skipped if it's still busy."""

    def __init__(self, clip, model):
        super().__init__(daemon=True)   # daemon: stops by itself when the app closes
        self.clip = clip
        self.model = model              # can be swapped while running (R key)
        self._crop = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.result = None              # (model used, ranked guesses)
        self.result_id = 0              # goes up by one with every new result
        self.ms = 0.0                   # how long the last picture took
        self.error = ""

    def submit(self, crop):
        with self._lock:
            self._crop = crop
        self._wake.set()

    def run(self):
        while True:
            self._wake.wait()
            with self._lock:
                crop, self._crop = self._crop, None
                self._wake.clear()
            if crop is None or crop.size == 0:
                continue
            model = self.model
            started = time.perf_counter()
            try:
                guesses = model.predict(self.clip.embed_images([crop])[0])
                self.error = ""
            except Exception as error:   # show the problem on screen instead of silently stopping
                guesses, self.error = [], f"Recognition error: {error}"
            self.ms = (time.perf_counter() - started) * 1000
            with self._lock:
                self.result = (model, guesses)
                self.result_id += 1


def average(history):
    """Average the last few answers: [(label, probability), ...] best first."""
    totals = {}
    for guesses in history:
        for label, probability in guesses:
            totals[label] = totals.get(label, 0.0) + probability / len(history)
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def headline(ranked):
    """The main answer to show on the box."""
    if not ranked:
        return "..."
    label, probability = ranked[0]
    if probability < NOT_SURE_BELOW:
        return "Not sure"
    if label == NO_ITEM:
        return "No item"
    return f"{label.replace('_', ' ')} {probability:.0%}"


def draw_panel(frame, crop_mode, ranked, ms, message):
    """Crop mode, top guesses and speed in the top-left corner; messages at the bottom."""
    lines = [f"Crop: {MODE_NAMES[crop_mode]}  (C)"]
    lines += [f"{label.replace('_', ' '):14s} {probability:4.0%}" for label, probability in ranked[:TOP_N]]
    if ms:
        lines.append(f"CLIP: {ms:.0f} ms per picture")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 60 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOUR, 2)
    if message:
        cv2.putText(frame, message, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNING_COLOUR, 2)


def main():
    model = ItemModel.load()
    if model is None:
        print(NOT_TRAINED)
        return
    print(f"Trained model loaded: {', '.join(model.labels)}")

    from .clip_model import Clip   # needs PyTorch, so only loaded once we know there's a model to use
    clip = Clip()                  # loads CLIP's picture encoder once
    worker = Worker(clip, model)
    worker.start()
    camera = Camera()
    tracker = HandTracker()        # finds your hand for the hand crop
    crop_mode = "auto"
    history = deque(maxlen=AVERAGE_OVER)
    seen_id = 0
    note, note_until = "", 0.0     # short messages after key presses
    try:
        for frame in camera.frames():
            hands = tracker.detect(frame) if crop_mode != "centre" else []
            box, source = choose_box(crop_mode, hands, frame)
            if box is not None:
                worker.submit(cut(frame, box))   # cut BEFORE drawing on the frame
            else:
                history.clear()

            # Pick up a new answer from the background thread (ignore answers from a model we just replaced).
            if worker.result_id != seen_id:
                seen_id = worker.result_id
                used_model, guesses = worker.result
                if used_model is worker.model and guesses:
                    history.append(guesses)
            ranked = average(history) if box is not None else []

            message = worker.error
            if not message and box is None:
                message = "Show your hand (crop mode: Hand only, press C to change)"
            if time.perf_counter() < note_until:
                message = note

            if box is not None:
                draw_box(frame, box, source, headline(ranked))
            draw_panel(frame, crop_mode, ranked, worker.ms, message)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey(1) waits 1 ms for a key press and lets the window refresh.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                crop_mode = next_mode(crop_mode)
                history.clear()
            elif key in (ord("r"), ord("R")):
                reloaded = ItemModel.load()
                if reloaded is not None:
                    worker.model = reloaded
                    history.clear()
                    note = f"Reloaded model: {', '.join(reloaded.labels)}"
                else:
                    note = NOT_TRAINED
                note_until = time.perf_counter() + 2
            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Always free the webcam and close the window, even if something went wrong.
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
