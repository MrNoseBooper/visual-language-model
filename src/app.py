"""Main app. Shows the live webcam feed, tracks your hands and names YOUR trained gestures.

Run with:  python src/app.py

Keys (click the video window first):
  R   reload your model (after running train.py again)
  Q   quit

One-hand and two-hand gestures are recognised automatically. When both hands are visible, the
two-hand model is asked first; if it doesn't see a two-hand gesture, each hand is named separately
by the one-hand model.
"""
import cv2

from camera import Camera
from classifier import GestureModel
from features import normalise, normalise_two
from hands import GestureSmoother, HandTracker, draw_hand, draw_label_above, pair_hands

WINDOW_NAME = "Gesture Recognizer (press Q to quit)"
BOTH_HANDS_COLOUR = (0, 255, 255)   # yellow label for two-hand gestures
NO_GESTURE_LABELS = {"none", "Not sure", "..."}   # these don't count as a two-hand gesture


def load_model():
    """Load your trained model, or None if it hasn't been trained yet."""
    model = GestureModel.load()
    if model is None:
        print("Gesture model: not trained yet")
    else:
        print(f"Gesture model loaded | one-hand: {'yes' if model.one_hand else 'no'} | "
              f"two-hand: {', '.join(model.two_handed_gestures) or 'none'}")
    return model


def label_text(shown, raw_label, score):
    """What to write on screen: the gesture, plus its score once it's stable."""
    text = shown.replace("_", " ")
    return f"{text} {score:.0%}" if shown == raw_label else text


def annotate(frame, hands, model, smoother):
    """Draw every hand with its gesture label. Returns a short status message (or "")."""
    if model is None:
        for hand in hands:
            draw_hand(frame, hand)
        return "No gesture model yet: run collect.py, then train.py, then press R"

    # Two-hand gesture first: if both hands are visible and the two-hand model is confident
    # about a real gesture, show one label over both hands instead of one per hand.
    pair = pair_hands(hands) if model.two_hand else None
    present = {hand.handedness for hand in hands} | ({"Both"} if pair else set())
    smoother.forget_missing(present)
    if pair:
        left, right = pair
        label, score = model.predict_two(normalise_two(
            left.world_landmarks, right.world_landmarks, left.landmarks, right.landmarks, left.aspect))
        shown = smoother.update("Both", label, score)
        if shown not in NO_GESTURE_LABELS:
            for hand in hands:
                draw_hand(frame, hand)
            draw_label_above(frame, hands, label_text(shown, label, score), BOTH_HANDS_COLOUR)
            return ""

    # Otherwise, one label per hand from the one-hand model.
    if model.one_hand is None:
        for hand in hands:
            draw_hand(frame, hand)
        return "No one-hand model: train.py needs a one-hand gesture + 'none'" if hands else ""
    for hand in hands:
        label, score = model.predict_one(normalise(hand.world_landmarks, hand.handedness))
        shown = smoother.update(hand.handedness, label, score)
        draw_hand(frame, hand, label_text(shown, label, score))
    return ""


def draw_message(frame, message):
    """Status message along the bottom of the window."""
    if message:
        cv2.putText(frame, message, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def main():
    camera = Camera()
    tracker = HandTracker()   # loads the hand model once, before the loop
    model = load_model()
    smoother = GestureSmoother()
    try:
        for frame in camera.frames():
            hands = tracker.detect(frame)
            message = annotate(frame, hands, model, smoother)

            draw_message(frame, message)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey(1) waits 1 ms for a key press and lets the window refresh.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                model = load_model()
                smoother = GestureSmoother()   # start fresh so old labels don't carry over
            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Always free the webcam and close the window, even if something went wrong.
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
