"""Quick check that Python, OpenCV, MediaPipe and the webcam all work.

Run with:  python src/check_setup.py
"""
import sys

import cv2
import mediapipe as mp

CAMERA_INDEX = 0  # 0 = the default (built-in) webcam

print(f"Python    : {sys.version.split()[0]}")
print(f"OpenCV    : {cv2.__version__}")
print(f"MediaPipe : {mp.__version__}")

# Try to open the webcam and grab one frame.
# CAP_DSHOW uses Windows' DirectShow backend, which usually opens faster on Windows.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
ok, frame = cap.read()
cap.release()

if ok:
    height, width = frame.shape[:2]
    print(f"webcam OK ({width}x{height})")
else:
    print("webcam FAILED: check that no other app (Teams, Zoom, Camera) is using it,")
    print("and that camera access is allowed in Windows Settings > Privacy & security > Camera.")
