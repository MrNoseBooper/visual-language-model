"""Hand gesture recogniser: one- and two-hand gestures you train yourself.

MediaPipe finds the 21 points on each hand; a scikit-learn model trained on your own samples names the gesture.

Everything about it lives in this folder:
  app.py         live recognition
  collect.py     record samples (no photos, just hand-point numbers)
  train.py       train the models
  features.py    turns hand points into numbers (normalise / normalise_two)
  classifier.py  loads the trained model for the app
  labels.txt     your gesture names, one per line
  data/          landmarks.csv: your recorded samples (not in git)
  models/        gesture_clf.joblib: the trained model (not in git)
  runs/          confusion matrices from training (not in git)

Use it through main.py:  python main.py gestures run | collect | train
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
NAME = "Hand gestures"
DESCRIPTION = "one- and two-hand gestures you train yourself (MediaPipe hand points + scikit-learn)"
# main.py runs app.py / collect.py / train.py's main() when you pick run / collect / train.
