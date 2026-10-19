# Hand Gesture Recognizer: notes for Claude

A Python webcam app that tracks my hands with MediaPipe and recognises gestures.
Stages: MediaPipe built-in gestures → custom static gestures (scikit-learn on normalised landmarks)
→ optional dynamic gestures (swipes, wave) → gestures trigger keyboard actions.

## Me and my machine
- I'm learning as I go: explain in plain words, step by step, and say what you changed and why.
- Windows, VS Code, PowerShell. Python 3.12 in .venv.
- Laptop: Intel Core Ultra 7 258V (Arc 140V iGPU + NPU). No NVIDIA GPU / no CUDA. Everything should run on CPU.

## Structure
src/        camera.py, hands.py, features.py, collect.py, train.py, dynamic.py, actions.py, app.py
data/       landmarks.csv, sequences/<label>/*.npy   (not in git)
models/     MediaPipe .task files and trained classifiers (not in git)
runs/       outputs: confusion matrices (not in git)
gestures.txt  one custom gesture label per line
actions.json  gesture -> action mapping

## Conventions
- Keep scripts small and commented; constants (thresholds, model paths, timings) at the top of each file.
- Frames are mirrored (selfie view) before MediaPipe detection. Despite MediaPipe's docs, its Left/Right labels
  came out reversed on this laptop, so hands.py swaps them (SWAP_HANDEDNESS). Hand.handedness = the user's real hand.
- One normalise() in features.py, used by collect, train and app. Never duplicate it.
- Load models once, never per frame. Heavy or network work goes in a background thread.
- Secrets (API keys) only from environment variables.
- Prefer permissive licences (MIT / Apache / BSD); ask before adding AGPL or research-only code or models.
- After each change, tell me exactly what command to run to test it.
