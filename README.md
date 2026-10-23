# visual-language-model

Webcam recognisers that run on a laptop CPU (no NVIDIA GPU needed), all started from one master file:

- **Hand gestures**: MediaPipe finds your hands; a small model trained on your own samples names one- and
  two-hand gestures (one-hand vs two-hand is worked out automatically).
- **Items**: names the item you show the camera, learned only from your own photos. CLIP's picture encoder
  turns each photo into numbers and a small model trained on your photos names them; no item names are
  built in, so it recognises nothing until you train it. It looks around your hand when one is visible,
  otherwise at the centre of the view.
- **Face tracer**: MediaPipe finds the points of your face and draws its lines live (outline, eyes, eyebrows,
  lips, irises). Nothing to train.

## Setup (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py check
```

Download MediaPipe's hand model into `common/models/`:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

For the face tracer, download MediaPipe's face model into `recognisers/faces/models/`:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

CLIP's picture encoder (one ~600 MB file) downloads by itself into `recognisers/items/models/clip/` the first
time you train the item recogniser.

## Use

```powershell
python main.py                     # menu: choose a recogniser, then run / collect / train
python main.py gestures collect    # record gesture samples
python main.py gestures train      # train the gesture model
python main.py gestures run        # live gesture recognition
python main.py items collect       # record item photos
python main.py items train         # train the item model
python main.py items run           # live item recognition
python main.py faces run           # live face tracing
```

Add `none` samples to every recogniser (relaxed hands / no item), so it learns when there's nothing to recognise.

**Keys in the live apps:** Q quit, R reload the trained model. Items also: C crop mode
(auto / hand / centre). **Collecting:** SPACE save, B burst mode, 1-9 switch label (items: C crop mode).

## Layout

```
main.py                   master file
common/                   shared: camera, hand tracker, training steps, label lists, setup check
recognisers/
  gestures/               everything about hand gestures: code, labels.txt, data/, models/, runs/
  items/                  everything about items: code, labels.txt, data/, models/, runs/
  faces/                  face tracer: app.py, models/
```

Recordings (`data/`), models (`models/`) and training outputs (`runs/`) stay on your computer and are not in git.
To add a recogniser, see `recognisers/__init__.py`.

## Licences

MediaPipe (Apache-2.0), CLIP (MIT), scikit-learn (BSD), PyTorch (BSD), Transformers (Apache-2.0).
