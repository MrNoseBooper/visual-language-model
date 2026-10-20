"""Names the item in a picture using the model train.py made from YOUR photos (models/probe.joblib).

There is no built-in knowledge of item names: until you collect photos and train, nothing is recognised.
Returns every label with a probability, best first: [("phone", 0.82), ("pen", 0.10), ...].
"""
from pathlib import Path

import joblib
import numpy as np

from . import FOLDER

LABELS_FILE = FOLDER / "labels.txt"
PROBE_PATH = FOLDER / "models" / "probe.joblib"
NO_ITEM = "none"   # the "nothing to recognise" label


class ItemModel:
    def __init__(self, path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.labels = bundle["labels"]

    @classmethod
    def load(cls, path=PROBE_PATH):
        """Load the trained model, or None if it doesn't exist yet."""
        return cls(path) if Path(path).exists() else None

    def predict(self, embedding):
        probabilities = self.model.predict_proba(embedding.reshape(1, -1))[0]
        order = np.argsort(probabilities)[::-1]
        return [(str(self.model.classes_[i]), float(probabilities[i])) for i in order]
