"""Training steps shared by every recogniser.

train_best() takes numbers X (one row per sample) and names y, and:
  1. splits them into train (70%), validation (15%) and test (15%) sets,
  2. tries three kinds of model and keeps the one that does best on the validation set,
  3. retrains that model on train + validation, then measures it on the test set
     (samples it has never seen), and saves a confusion matrix picture.
"""
from collections import Counter

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

RANDOM_SEED = 42           # fixed, so the split and results are the same every run
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15
MIN_SAMPLES_PER_CLASS = 10      # fewer than this and the split can't work
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


def check_counts(y, no_label="none"):
    """Print samples per class. Returns False if there isn't enough data to train."""
    counts = Counter(y.tolist())   # .tolist() gives plain names for printing
    print(f"Samples per class: {dict(sorted(counts.items()))}")
    if len(counts) < 2:
        print(f"  Need at least 2 classes to train (e.g. a label + '{no_label}'). Skipping.")
        return False
    too_few = [label for label, count in counts.items() if count < MIN_SAMPLES_PER_CLASS]
    if too_few:
        print(f"  These need at least {MIN_SAMPLES_PER_CLASS} samples each: {too_few}. Skipping.")
        return False
    few = [label for label, count in counts.items() if count < FEW_SAMPLES_WARNING]
    if few:
        print(f"  Warning: under {FEW_SAMPLES_WARNING} samples for {few}; results may be unreliable.")
    if no_label not in counts:
        print(f"  Warning: no '{no_label}' samples. The model will always pick one of your labels, even when "
              "there's nothing to recognise.")
    return True


def save_confusion_matrix(path, title, model, X_test, y_test):
    """Picture showing, for each real label (rows), what the model guessed (columns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues",
                                          xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_best(name, X, y, runs_dir, no_label="none"):
    """Pick and train the best model for X, y. Returns (model, test accuracy), or (None, None)."""
    print(f"\n=== {name} model ===")
    if len(y) == 0:
        print("No samples. Skipping.")
        return None, None
    X, y = np.array(X, dtype=np.float32), np.array(y)
    if not check_counts(y, no_label):
        return None, None

    # Split: first take off 30%, then cut that in half for validation and test.
    # "stratify" keeps the same mix of labels in every part.
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
    picture = save_confusion_matrix(runs_dir / f"confusion_{name}.png", f"{name} model: test set",
                                    best_model, X_test, y_test)
    print(f"Confusion matrix saved to {picture}")
    return best_model, test_accuracy
