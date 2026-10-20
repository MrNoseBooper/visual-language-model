"""Item recogniser: names the item you show the camera, learned ONLY from your own photos.

CLIP's picture encoder turns each photo into 512 numbers; a small model trained on your photos
(data/<label>/) learns which numbers belong to which item. No item names are built in: nothing is
recognised until you collect photos and train.
It looks around your hand when one is visible, otherwise at the centre of the view (press C to switch).

Everything about it lives in this folder:
  app.py           live recognition
  collect.py       record photos of your items
  train.py         train on your photos
  clip_model.py    loads CLIP's picture encoder: turns pictures into numbers ("embeddings")
  predictor.py     loads your trained model and names the item
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
DESCRIPTION = "names items you show the camera, learned only from your own photos (CLIP picture encoder)"
# main.py runs app.py / collect.py / train.py's main() when you pick run / collect / train.
