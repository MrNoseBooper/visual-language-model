"""Loads the trained gesture model file (made by train.py) and uses it to name gestures.

The file holds two models inside it:
- one_hand: names the gesture of a single hand (63 numbers).
- two_hand: when both hands are visible, says which two-hand gesture they make, or "none" (210 numbers).
Either can be missing (None) if there wasn't data to train it.

Used by app.py. Kept separate from train.py so the app doesn't need any of the training code.
"""
from pathlib import Path

import joblib
import numpy as np

from . import FOLDER

MODEL_PATH = FOLDER / "models" / "gesture_clf.joblib"


class GestureModel:
    def __init__(self, path):
        bundle = joblib.load(path)
        self.one_hand = bundle["one_hand"]
        self.two_hand = bundle["two_hand"]
        self.two_handed_gestures = bundle["two_handed_gestures"]

    @classmethod
    def load(cls, path=MODEL_PATH):
        """Load the model file, or return None if it doesn't exist yet (not trained)."""
        return cls(path) if Path(path).exists() else None

    def predict_one(self, features):
        """(gesture name, probability 0..1) for one hand's 63 numbers."""
        return _predict(self.one_hand, features)

    def predict_two(self, features):
        """(two-hand gesture name or "none", probability 0..1) for a pair's 210 numbers."""
        return _predict(self.two_hand, features)


def _predict(model, features):
    probabilities = model.predict_proba(np.asarray(features).reshape(1, -1))[0]
    best = int(np.argmax(probabilities))
    return str(model.classes_[best]), float(probabilities[best])
