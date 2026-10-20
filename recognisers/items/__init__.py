"""Item recogniser: names the item you show the camera, using CLIP (a vision-language model).

It works with NO training ("zero-shot"): CLIP compares the picture with each name in labels.txt.
After you collect photos of your own items and train, a small model learns them and gets more accurate.
It looks around your hand when one is visible, otherwise at the centre of the view (press C to switch).

Everything about it lives in this folder:
  app.py           live recognition
  collect.py       record photos of your items
  train.py         train on your photos
  clip_model.py    loads CLIP and turns pictures/text into numbers ("embeddings")
  predictor.py     zero-shot and trained ways of naming the item
  crop.py          picks which part of the picture to look at (hand or centre)
  labels.txt       your item names, one per line
  data/            <label>/*.jpg: your photos (not in git)
  models/          clip/ (downloaded CLIP model) and probe.joblib (trained model) (not in git)
  runs/            embeddings cache and confusion matrices from training (not in git)

Use it through main.py:  python main.py items run | collect | train
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
NAME = "Items"
DESCRIPTION = "names items you show the camera (CLIP vision-language model; works untrained, better trained)"
# main.py runs app.py / collect.py / train.py's main() when you pick run / collect / train.
