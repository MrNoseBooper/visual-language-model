# Hand Gesture Recognizer: notes for Claude

A Python webcam app that tracks my hands with MediaPipe and recognises gestures.
No MediaPipe built-in gestures: MediaPipe's Hand Landmarker only finds the hands; all gestures come from my own model.
Stages: custom static gestures (scikit-learn on normalised landmarks)
→ optional dynamic gestures (swipes, wave) → gestures trigger keyboard actions.

## Me and my machine
- I'm learning as I go: explain in plain words, step by step, and say what you changed and why.
- Windows, VS Code, PowerShell. Python 3.12 in .venv.
- Laptop: Intel Core Ultra 7 258V (Arc 140V iGPU + NPU). No NVIDIA GPU / no CUDA. Everything should run on CPU.

## Structure
src/        camera.py, hands.py, features.py, collect.py, train.py, classifier.py (loads trained models),
            dynamic.py, actions.py, app.py
data/       landmarks.csv (all samples; "hands" column = 1 or 2), sequences/<label>/*.npy   (not in git)
models/     MediaPipe .task files and trained classifiers (not in git)
runs/       outputs: confusion matrices (not in git)
gestures.txt  one custom gesture label per line (one-hand and two-hand together)
actions.json  gesture -> action mapping

## Conventions
- Keep scripts small and commented; constants (thresholds, model paths, timings) at the top of each file.
- Frames are mirrored (selfie view) before MediaPipe detection. Despite MediaPipe's docs, its Left/Right labels
  came out reversed on this laptop, so hands.py swaps them (SWAP_HANDEDNESS). Hand.handedness = the user's real hand.
- One normalise() (one hand, 63 numbers) and one normalise_two() (two hands, 210 numbers) in features.py, used by
  collect, train and app. Never duplicate them. Two-hand pairs are ordered with hands.pair_hands().
- One-hand vs two-hand is automatic, never a user switch: collect saves 1- or 2-hand samples by how many hands
  are visible; train marks a gesture two-handed if most samples are 2-hand, and saves a one-hand and a two-hand
  model together in models/gesture_clf.joblib; app tries the two-hand model first when both hands are visible.
- Load models once, never per frame. Heavy or network work goes in a background thread.
- Secrets (API keys) only from environment variables.
- Prefer permissive licences (MIT / Apache / BSD); ask before adding AGPL or research-only code or models.
- After each change, tell me exactly what command to run to test it.
