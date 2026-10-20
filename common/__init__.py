"""Shared building blocks used by every recogniser.

camera.py       webcam frames (mirrored) and FPS counter
hands.py        MediaPipe hand tracking, label smoothing, drawing helpers
training.py     split data, compare scikit-learn models, report accuracy, save a confusion matrix
check_setup.py  prints library versions and tests the webcam
models/         hand_landmarker.task (MediaPipe's hand model, not in git)
"""
