"""Main app. For now it just shows the live webcam feed with an FPS counter.

Run with:  python src/app.py
Press Q to quit.
"""
import cv2

from camera import Camera

WINDOW_NAME = "Gesture Recognizer (press Q to quit)"


def main():
    camera = Camera()
    try:
        for frame in camera.frames():
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
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
