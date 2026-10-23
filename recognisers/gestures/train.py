"""Train the gesture model on the samples you recorded with collect.py.

Run with:  python main.py gestures train

Reads data/landmarks.csv and works out by itself which gestures are two-handed: a gesture is
two-handed if almost all (TWO_HAND_SHARE) of its samples were recorded with both hands in view. It then trains two models
and saves them together in models/gesture_clf.joblib:
  - one-hand model: names a single hand's gesture (from the 1-hand samples of one-hand gestures + "none").
  - two-hand model: when both hands are visible, says which two-hand gesture they make, or "none".
    2-hand samples of one-hand gestures (e.g. other hand relaxed) are taught as "none" here, so the
    app falls back to naming each hand separately.

Each model is picked and tested by common/training.py (train / validation / test split, three model types,
confusion matrix saved to runs/).
"""
import csv
from collections import Counter

import joblib

from common.training import train_best

from . import FOLDER
from .classifier import MODEL_PATH
from .collect import DATA_FILE
from .features import NUM_FEATURES, NUM_TWO_HAND_FEATURES

RUNS_DIR = FOLDER / "runs"
NO_GESTURE = "none"
# A gesture is two-handed only if at least this share of its samples have both hands in view.
# Kept high so a one-hand gesture you also recorded with both hands up (e.g. two open palms) stays one-handed.
TWO_HAND_SHARE = 0.8


def load_samples(path):
    """Read data/landmarks.csv into a list of (gesture name, number of hands, numbers)."""
    samples = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            hands = int(row["hands"])
            count = NUM_FEATURES if hands == 1 else NUM_TWO_HAND_FEATURES
            samples.append((row["label"], hands, [float(row[f"f{i}"]) for i in range(count)]))
    return samples


def find_two_handed(samples):
    """Gestures (other than "none") recorded almost only with two hands. Prints a table of what it found."""
    counts = Counter((label, hands) for label, hands, _ in samples)
    labels = sorted({label for label, _, _ in samples})
    two_handed = {label for label in labels
                  if label != NO_GESTURE
                  and counts[(label, 2)] >= TWO_HAND_SHARE * (counts[(label, 1)] + counts[(label, 2)])}

    print(f"  {'gesture':15s} {'1-hand':>7s} {'2-hand':>7s}   type")
    for label in labels:
        kind = "(no gesture)" if label == NO_GESTURE else "TWO-hand" if label in two_handed else "one-hand"
        print(f"  {label:15s} {counts[(label, 1)]:7d} {counts[(label, 2)]:7d}   {kind}")
    return two_handed


def main():
    if not DATA_FILE.exists():
        print(f"No data yet ({DATA_FILE}). Record some with:  python main.py gestures collect")
        return

    samples = load_samples(DATA_FILE)
    print(f"Loaded {len(samples)} samples from {DATA_FILE.name}:")
    two_handed = find_two_handed(samples)

    # One-hand model: 1-hand samples, except those of two-hand gestures (half a gesture isn't useful).
    one = [(label, x) for label, hands, x in samples if hands == 1 and label not in two_handed]
    skipped = sum(1 for label, hands, _ in samples if hands == 1 and label in two_handed)
    if skipped:
        print(f"\n(Ignoring {skipped} 1-hand samples of two-hand gestures.)")
    one_model, one_accuracy = train_best("one-hand", [x for _, x in one], [label for label, _ in one],
                                         RUNS_DIR, NO_GESTURE)

    # Two-hand model: every 2-hand sample. Anything that isn't a two-hand gesture is taught as "none".
    two_model, two_accuracy = None, None
    if two_handed:
        two = [(label if label in two_handed else NO_GESTURE, x) for label, hands, x in samples if hands == 2]
        two_model, two_accuracy = train_best("two-hand", [x for _, x in two], [label for label, _ in two],
                                             RUNS_DIR, NO_GESTURE)
        if two_model is not None and NO_GESTURE not in two_model.classes_:
            print("Tip: also record 'none' (or one-hand gestures) with BOTH hands in view, so two relaxed hands\n"
                  "aren't mistaken for a two-hand gesture.")
    else:
        print("\nNo two-hand gestures recorded, so no two-hand model (that's fine).")

    print("\n=== Summary ===")
    for name, accuracy in [("one-hand", one_accuracy), ("two-hand", two_accuracy)]:
        print(f"  {name:9s} " + (f"test accuracy {accuracy:.1%}" if accuracy is not None else "not trained"))
    if one_model is None and two_model is None:
        print("Nothing saved.")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "one_hand": one_model,
        "two_hand": two_model,
        "two_handed_gestures": sorted(two_handed),
        "test_accuracy": {"one_hand": one_accuracy, "two_hand": two_accuracy},
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Try it live:  python main.py gestures run   (press R in the app if it's already open)")
