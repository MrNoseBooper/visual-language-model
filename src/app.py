"""Main app. Shows the live webcam feed with an FPS counter and tracks your hands.

Run with:  python src/app.py
Press Q to quit.
"""
import cv2

from camera import Camera
from hands import HandTracker, draw_hand

WINDOW_NAME = "Gesture Recognizer (press Q to quit)"


def main():
    camera = Camera()
    tracker = HandTracker()   # loads the hand model once, before the loop
    try:
        for frame in camera.frames():
            # Find hands and draw each one's skeleton. With no hands, this list is just empty.
            for hand in tracker.detect(frame):
                draw_hand(frame, hand)

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
