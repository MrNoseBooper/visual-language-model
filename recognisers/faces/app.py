"""Live face tracing: draws the outline of your face, eyes, eyebrows, lips and irises.

Run with:  python main.py faces run

Keys (click the video window first):
  Q   quit

Model file:  recognisers/faces/models/face_landmarker.task
Downloaded from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections, RunningMode

from common.camera import Camera

from . import FOLDER

MODEL_PATH = FOLDER / "models" / "face_landmarker.task"
MAX_FACES = 2
WINDOW_NAME = "Face Tracer (press Q to quit)"
# Which points to join: outlines of face, eyes, eyebrows and lips, plus both irises.
# Swap in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION to see the full triangle mesh instead.
LINES = [*FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
         *FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
         *FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS]
LINE_COLOUR = (0, 255, 0)   # green (OpenCV uses Blue, Green, Red order)


def draw_face(frame, points):
    """Join the face points with lines. points: MediaPipe's landmarks, x and y as 0..1 of the image."""
    height, width = frame.shape[:2]
    pixels = [(int(p.x * width), int(p.y * height)) for p in points]
    for line in LINES:
        cv2.line(frame, pixels[line.start], pixels[line.end], LINE_COLOUR, 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face model not found at {MODEL_PATH}. Download it from the URL at the top of "
                                "recognisers/faces/app.py.")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,   # follows faces from frame to frame: smoother and faster
        num_faces=MAX_FACES,
    )
    landmarker = FaceLandmarker.create_from_options(options)   # load once, before the loop
    camera = Camera()
    last_ms = -1
    try:
        for frame in camera.frames():
            # OpenCV gives BGR colour order; MediaPipe wants RGB.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_ms = max(int(time.perf_counter() * 1000), last_ms + 1)   # VIDEO mode needs rising timestamps
            result = landmarker.detect_for_video(image, last_ms)

            for points in result.face_landmarks:
                draw_face(frame, points)
            if not result.face_landmarks:
                cv2.putText(frame, "No face in view", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            camera.draw_fps(frame)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            # Also stop if the window was closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Always free the webcam and close the window, even if something went wrong.
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()
