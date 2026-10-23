"""Face tracer: draws the lines of your face live. Nothing is trained or named yet.

MediaPipe's Face Landmarker finds 478 points on each face; app.py joins them into the outline of the face,
eyes, eyebrows, lips and irises.

Everything about it lives in this folder:
  app.py      live tracing
  models/     face_landmarker.task: MediaPipe's face model (not in git)

Use it through main.py:  python main.py faces run
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
NAME = "Face tracer"
DESCRIPTION = "draws the lines of your face live (MediaPipe face points, no training)"
