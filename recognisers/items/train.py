"""Train the item model on the photos you recorded with collect.py.

Run with:  python main.py items train

  1. Finds every photo in data/<label>/.
  2. Turns each photo into CLIP's 512 numbers. These are cached in runs/embeddings.npz, so next time
     only new or changed photos need CLIP (training again is then very quick).
  3. Picks and tests the best small model on those numbers (common/training.py: train / validation /
     test split, three model types, confusion matrix saved to runs/).
  4. Saves it to models/probe.joblib, which the live app uses.

This "CLIP numbers + small model" approach is called a linear probe. CLIP's picture encoder knows no item
names, but it describes pictures well, so a few dozen of your photos per item are often enough.
"""
import joblib
import numpy as np

from common.training import train_best

from . import FOLDER
from .clip_model import MODEL_NAME
from .collect import DATA_DIR, IMAGE_TYPES
from .predictor import NO_ITEM, PROBE_PATH

RUNS_DIR = FOLDER / "runs"
CACHE_FILE = RUNS_DIR / "embeddings.npz"


def find_photos():
    """[(path, label)] for every photo in data/<label>/."""
    photos = []
    if DATA_DIR.exists():
        for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
            for pattern in IMAGE_TYPES:
                photos += [(path, folder.name) for path in sorted(folder.glob(pattern))]
    return photos


def cache_key(path):
    """Photo name + when it was last changed: if either changes, CLIP runs on it again."""
    return f"{path.relative_to(DATA_DIR).as_posix()}|{path.stat().st_mtime_ns}"


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    saved = np.load(CACHE_FILE)
    return dict(zip(saved["keys"].tolist(), saved["embeddings"]))


def save_cache(cache):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(cache)
    np.savez(CACHE_FILE, keys=np.array(keys), embeddings=np.array([cache[key] for key in keys]))


def read_photo(path):
    """Read a photo (fromfile + imdecode instead of imread, so any folder name works on Windows)."""
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def embed_new_photos(photos, cache):
    """Run CLIP on photos that aren't in the cache yet. Returns how many were added."""
    new = [path for path, _ in photos if cache_key(path) not in cache]
    if not new:
        return 0
    from .clip_model import BATCH_SIZE, Clip
    clip = Clip()
    print(f"Running CLIP on {len(new)} new photos ...")
    for start in range(0, len(new), BATCH_SIZE):
        batch = [(path, read_photo(path)) for path in new[start:start + BATCH_SIZE]]
        readable = [(path, image) for path, image in batch if image is not None]
        for path, image in batch:
            if image is None:
                print(f"  Skipping unreadable photo: {path}")
        if readable:
            embeddings = clip.embed_images([image for _, image in readable])
            for (path, _), embedding in zip(readable, embeddings):
                cache[cache_key(path)] = embedding
        print(f"  {min(start + BATCH_SIZE, len(new))}/{len(new)}")
    return len(new)


def main():
    photos = find_photos()
    if not photos:
        print(f"No photos yet ({DATA_DIR}). Record some with:  python main.py items collect")
        return

    labels = sorted({label for _, label in photos})
    print(f"Found {len(photos)} photos:")
    for label in labels:
        print(f"  {label:15s} {sum(1 for _, name in photos if name == label)}")

    cache = load_cache()
    added = embed_new_photos(photos, cache)
    # Keep only photos that still exist, so deleted photos drop out of the cache.
    current = {cache_key(path): label for path, label in photos if cache_key(path) in cache}
    cache = {key: cache[key] for key in current}
    save_cache(cache)
    print(f"CLIP numbers: {added} new, {len(cache) - added} reused from the cache")

    model, accuracy = train_best("items", list(cache.values()), list(current.values()), RUNS_DIR, NO_ITEM)
    if model is None:
        print("Nothing saved.")
        return

    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": sorted(set(current.values())),
                 "clip_model": MODEL_NAME, "test_accuracy": accuracy}, PROBE_PATH)
    print(f"Model saved to {PROBE_PATH}")
    print("Try it live:  python main.py items run   (press R in the app if it's already open)")
