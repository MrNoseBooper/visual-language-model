"""Train the gesture model on the samples you recorded with collect.py.

Run with:  python src/train.py

Reads data/landmarks.csv and works out by itself which gestures are two-handed: a gesture is
two-handed if most of its samples were recorded with both hands in view. It then trains two models
and saves them together in models/gesture_clf.joblib:
  - one-hand model: names a single hand's gesture (from the 1-hand samples of one-hand gestures + "none").
  - two-hand model: when both hands are visible, says which two-hand gesture they make, or "none".
    2-hand samples of one-hand gestures (e.g. other hand relaxed) are taught as "none" here, so the
    app falls back to naming each hand separately.

For each model it:
  1. splits the samples into train (70%), validation (15%) and test (15%) sets,
  2. tries three kinds of model and keeps the one that does best on the validation set,
  3. retrains that model on train + validation, then measures it on the test set
     (samples it has never seen), and saves a confusion matrix picture to runs/.
"""
import csv
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")   # save pictures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifier import MODEL_PATH
from features import NUM_FEATURES, NUM_TWO_HAND_FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "landmarks.csv"
RUNS_DIR = ROOT / "runs"
NO_GESTURE = "none"
RANDOM_SEED = 42           # fixed, so the split and results are the same every run
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15
MIN_SAMPLES_PER_GESTURE = 10    # fewer than this and the split can't work
FEW_SAMPLES_WARNING = 30        # fewer than this and results will be unreliable


def candidate_models():
    """The models to compare. StandardScaler rescales the numbers, which LR and MLP need."""
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "MLP (small neural net)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_SEED),
        ),
    }


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
    """Gestures (other than "none") recorded mostly with two hands. Prints a table of what it found."""
    counts = Counter((label, hands) for label, hands, _ in samples)
    labels = sorted({label for label, _, _ in samples})
    two_handed = {label for label in labels
                  if label != NO_GESTURE and counts[(label, 2)] > counts[(label, 1)]}

    print(f"  {'gesture':15s} {'1-hand':>7s} {'2-hand':>7s}   type")
    for label in labels:
        kind = "(no gesture)" if label == NO_GESTURE else "TWO-hand" if label in two_handed else "one-hand"
        print(f"  {label:15s} {counts[(label, 1)]:7d} {counts[(label, 2)]:7d}   {kind}")
    return two_handed


def check_counts(y):
    """Print samples per class. Returns False if there isn't enough data to train."""
    counts = Counter(y.tolist())   # .tolist() gives plain names for printing
    print(f"Samples per class: {dict(sorted(counts.items()))}")
    if len(counts) < 2:
        print("  Need at least 2 classes to train (e.g. a gesture + 'none'). Skipping.")
        return False
    too_few = [label for label, count in counts.items() if count < MIN_SAMPLES_PER_GESTURE]
    if too_few:
        print(f"  These need at least {MIN_SAMPLES_PER_GESTURE} samples each: {too_few}. Skipping.")
        return False
    few = [label for label, count in counts.items() if count < FEW_SAMPLES_WARNING]
    if few:
        print(f"  Warning: under {FEW_SAMPLES_WARNING} samples for {few}; results may be unreliable.")
    if NO_GESTURE not in counts:
        print(f"  Warning: no '{NO_GESTURE}' samples. The model will think every hand is making one of your gestures.")
    return True


def save_confusion_matrix(name, model, X_test, y_test):
    """Picture showing, for each real gesture (rows), what the model guessed (columns)."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"confusion_{name}.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues",
                                          xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name} model: test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_model(name, X, y):
    """Pick and train the best model for X, y. Returns (model, test accuracy), or (None, None)."""
    print(f"\n=== {name} model ===")
    if len(y) == 0:
        print("No samples. Skipping.")
        return None, None
    X, y = np.array(X, dtype=np.float32), np.array(y)
    if not check_counts(y):
        return None, None

    # Split: first take off 30%, then cut that in half for validation and test.
    # "stratify" keeps the same mix of gestures in every part.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=VALIDATION_SHARE + TEST_SHARE, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=TEST_SHARE / (VALIDATION_SHARE + TEST_SHARE),
        stratify=y_rest, random_state=RANDOM_SEED)
    print(f"Train {len(y_train)} | validation {len(y_val)} | test {len(y_test)} samples")

    # Try each model and score it on the validation set.
    print("Validation accuracy:")
    best_name, best_score = None, -1.0
    for model_name, model in candidate_models().items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_val, model.predict(X_val))
        print(f"  {model_name:24s} {score:.1%}")
        if score > best_score:
            best_name, best_score = model_name, score
    print(f"Best: {best_name}")

    # Retrain the best kind of model on train + validation (more data), then test it once.
    best_model = candidate_models()[best_name]
    best_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nTEST accuracy: {test_accuracy:.1%}   (on {len(y_test)} samples it never saw)")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print(f"Confusion matrix saved to {save_confusion_matrix(name, best_model, X_test, y_test)}")
    return best_model, test_accuracy


def main():
    if not DATA_FILE.exists():
        print(f"No data yet ({DATA_FILE}). Record some with:  python src/collect.py")
        return

    samples = load_samples(DATA_FILE)
    print(f"Loaded {len(samples)} samples from {DATA_FILE.name}:")
    two_handed = find_two_handed(samples)

    # One-hand model: 1-hand samples, except those of two-hand gestures (half a gesture isn't useful).
    one = [(label, x) for label, hands, x in samples if hands == 1 and label not in two_handed]
    skipped = sum(1 for label, hands, _ in samples if hands == 1 and label in two_handed)
    if skipped:
        print(f"\n(Ignoring {skipped} 1-hand samples of two-hand gestures.)")
    one_model, one_accuracy = train_model("one-hand", [x for _, x in one], [label for label, _ in one])

    # Two-hand model: every 2-hand sample. Anything that isn't a two-hand gesture is taught as "none".
    two_model, two_accuracy = None, None
    if two_handed:
        two = [(label if label in two_handed else NO_GESTURE, x) for label, hands, x in samples if hands == 2]
        two_model, two_accuracy = train_model("two-hand", [x for _, x in two], [label for label, _ in two])
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

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "one_hand": one_model,
        "two_hand": two_model,
        "two_handed_gestures": sorted(two_handed),
        "test_accuracy": {"one_hand": one_accuracy, "two_hand": two_accuracy},
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Try it live:  python src/app.py   (press R in the app if it's already open)")


if __name__ == "__main__":
    main()
