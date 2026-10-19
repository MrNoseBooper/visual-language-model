"""Main app. Shows the live webcam feed, tracks your hands and names their built-in gestures.

Run with:  python src/app.py
Press Q to quit.
"""
import cv2

from camera import Camera
from hands import GestureSmoother, HandTracker, draw_hand

WINDOW_NAME = "Gesture Recognizer (press Q to quit)"


def main():
    camera = Camera()
    tracker = HandTracker()   # loads the gesture model once, before the loop
    smoother = GestureSmoother()
    try:
        for frame in camera.frames():
            hands = tracker.detect(frame)
            smoother.forget_missing({hand.handedness for hand in hands})

            for hand in hands:
                shown = smoother.update(hand.handedness, hand.gesture, hand.gesture_score)
                # Add the score, unless we're unsure or still waiting for a stable answer.
                if shown == hand.gesture:
                    text = f"{shown.replace('_', ' ')} {hand.gesture_score:.0%}"
                else:
                    text = shown.replace("_", " ")
                draw_hand(frame, hand, text)

            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey(1) waits 1 ms for a key press and lets the window refresh.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
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
