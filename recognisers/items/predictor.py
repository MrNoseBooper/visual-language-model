"""Two ways of naming the item in a picture, both starting from CLIP's 512 picture numbers:

  ZeroShot  no training: compares the picture with "a photo of a <label>" for every label in labels.txt.
  Trained   the model train.py made from your own photos (models/probe.joblib). Usually more accurate.

Both return every label with a probability, best first: [("phone", 0.82), ("mug", 0.10), ...].
"""
from pathlib import Path

import joblib
import numpy as np

from . import FOLDER
from .clip_model import PROMPT

LABELS_FILE = FOLDER / "labels.txt"
PROBE_PATH = FOLDER / "models" / "probe.joblib"
NO_ITEM = "none"   # the "nothing to recognise" label: only the trained model can learn it


def _ranked(labels, probabilities):
    order = np.argsort(probabilities)[::-1]
    return [(str(labels[i]), float(probabilities[i])) for i in order]


class ZeroShot:
    name = "Zero-shot (no training)"

    def __init__(self, clip, labels):
        # "none" can't be described as "a photo of a none", so zero-shot leaves it out and
        # relies on the "Not sure" threshold in app.py instead.
        self.labels = [label for label in labels if label != NO_ITEM]
        self.logit_scale = clip.logit_scale
        prompts = [PROMPT.format(label.replace("_", " ")) for label in self.labels]
        self.text = clip.embed_texts(prompts) if prompts else None   # worked out once, not per frame

    def predict(self, embedding):
        if self.text is None:
            return []
        scores = self.logit_scale * (self.text @ embedding)    # similarity with each label's text
        probabilities = np.exp(scores - scores.max())
        return _ranked(self.labels, probabilities / probabilities.sum())


class Trained:
    name = "Trained on your photos"

    def __init__(self, path):
        self.model = joblib.load(path)["model"]

    @classmethod
    def load(cls, path=PROBE_PATH):
        """Load the trained model, or None if it doesn't exist yet."""
        return cls(path) if Path(path).exists() else None

    def predict(self, embedding):
        probabilities = self.model.predict_proba(embedding.reshape(1, -1))[0]
        return _ranked(self.model.classes_, probabilities)
