# Visual recognisers: notes for Claude

Python webcam apps on one master file. Two recognisers, each in its own folder with everything about it:
- **gestures**: MediaPipe Hand Landmarker finds the hands; my own scikit-learn models name one- and two-hand
  gestures. No MediaPipe built-in gestures.
- **items**: names the item shown, learned ONLY from my own photos: CLIP's picture encoder (openai/clip-vit-base-patch32,
  vision half only) turns photos into 512 numbers, a linear probe trained on my photos names them. NO zero-shot / no
  text prompts / no built-in item names (my decision). Looks around the hand if visible, else the centre (C key).
Later stages (gestures): optional dynamic gestures (swipes, wave) → gestures trigger keyboard actions.

## Me and my machine
- I'm learning as I go: explain in plain words, step by step, and say what you changed and why.
- Windows, VS Code, PowerShell. Python 3.12 in .venv. I install requirements myself.
- Laptop: Intel Core Ultra 7 258V (Arc 140V iGPU + NPU). No NVIDIA GPU / no CUDA. Everything should run on CPU.

## Structure
main.py                 master file: menu, or `python main.py <gestures|items> <run|collect|train>`, `python main.py check`
common/                 shared: camera.py, hands.py (tracker, LabelSmoother, drawing), training.py (train_best),
                        labels.py (labels.txt helpers), check_setup.py, models/hand_landmarker.task
recognisers/__init__.py RECOGNISERS registry (key -> package), ACTION_MODULES (run->app, collect, train)
tests/                  unittest checks: `python -m unittest discover tests`
recognisers/<name>/     __init__.py (FOLDER, NAME, DESCRIPTION only), app.py / collect.py / train.py (each has main()),
                        labels.txt, data/, models/, runs/  (data/models/runs not in git)
  gestures/             + features.py (normalise / normalise_two), classifier.py; data/landmarks.csv,
                        models/gesture_clf.joblib
  items/                + clip_model.py, predictor.py (ItemModel), crop.py; data/<label>/*.jpg,
                        models/clip/ (HF cache), models/probe.joblib, runs/embeddings.npz

## Conventions
- Run everything through main.py (imports are package-style: `from common.x import ...`, `from .x import ...`).
- Never define functions in a recogniser's __init__.py with the same name as one of its modules (e.g. train()):
  importing the module replaces the function. main.py imports app/collect/train modules and calls main().
- Keep scripts small and commented; constants (thresholds, model paths, timings) at the top of each file.
- Frames are mirrored (selfie view) before MediaPipe detection. Despite MediaPipe's docs, its Left/Right labels
  came out reversed on this laptop, so hands.py swaps them (SWAP_HANDEDNESS). Hand.handedness = the user's real hand.
- Gestures: one normalise() (63 numbers) and one normalise_two() (210 numbers) in features.py, used by collect,
  train and app. Never duplicate them. Two-hand pairs are ordered with common.hands.pair_hands().
- Gestures: one-hand vs two-hand is automatic, never a user switch: collect saves 1- or 2-hand samples by how many
  hands are visible; train marks a gesture two-handed if >= 80% of samples are 2-hand (TWO_HAND_SHARE) and saves a
  one-hand and a two-hand model together; app tries the two-hand model first when both hands are visible.
- Items: "none" is the no-item label. The app refuses to start without a trained model. CLIP runs in a background thread in
  the app; collect.py never imports torch. Crop BEFORE drawing on the frame.
- Load models once, never per frame. Heavy or network work goes in a background thread.
- Secrets (API keys) only from environment variables.
- Prefer permissive licences (MIT / Apache / BSD); ask before adding AGPL or research-only code or models.
- After each change, tell me exactly what command to run to test it.
