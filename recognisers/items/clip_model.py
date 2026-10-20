"""CLIP, the vision-language model behind the item recogniser.

CLIP turns a picture into 512 numbers (an "embedding"), and a piece of text into 512 numbers too.
A picture and a text that mean the same thing get similar numbers. That lets us:
  - name an item with NO training, by comparing the picture with "a photo of a phone", "a photo of a mug", ...
  - train a small, fast model on the picture numbers of your own photos (train.py).

The model (about 600 MB) is downloaded from Hugging Face the first time and kept in models/clip/.
Licence: MIT.
"""
import cv2
import numpy as np

from . import FOLDER

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = FOLDER / "models" / "clip"
PROMPT = "a photo of a {}"    # how each label is turned into text for zero-shot
BATCH_SIZE = 32               # pictures processed at once during training


class Clip:
    def __init__(self):
        print(f"Loading CLIP ({MODEL_NAME}). The first time, it downloads about 600 MB into {CACHE_DIR} ...")
        # Imported here, not at the top, so the rest of the app starts fast and collect.py never needs PyTorch.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        # Load the model once, never per frame. eval() switches off training-only behaviour.
        self.model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        # How strongly CLIP separates good and bad matches (used to turn similarities into percentages).
        self.logit_scale = float(self.model.logit_scale.exp())
        print("CLIP loaded.")

    def embed_images(self, images):
        """512 numbers for each picture (OpenCV BGR images), scaled to length 1. Shape (n, 512)."""
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        chunks = []
        for start in range(0, len(rgb), BATCH_SIZE):
            inputs = self.processor(images=rgb[start:start + BATCH_SIZE], return_tensors="pt")
            with self.torch.no_grad():   # no training here, so skip the extra bookkeeping
                pooled = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                chunks.append(self.model.visual_projection(pooled).numpy())
        return _unit_length(np.concatenate(chunks))

    def embed_texts(self, texts):
        """512 numbers for each piece of text, scaled to length 1. Shape (n, 512)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            pooled = self.model.text_model(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"]).pooler_output
            return _unit_length(self.model.text_projection(pooled).numpy())


def _unit_length(vectors):
    """Scale each row to length 1, so comparing two rows is a simple dot product (cosine similarity)."""
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
